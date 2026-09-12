import json
import time
import zipfile
from pathlib import Path

import requests

from config.mineru_config import mineru_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError, PdfConversionError
from processor.import_processor.state import ImportGraphState


class NodePDFToMD(BaseNode):
    """
    PDF 转 Markdown 节点：PDF结构化解析
    """

    name = "node_pdf_to_md"

    def process(self, state: ImportGraphState):
        #参数检查
        pdf_path_obj,output_dir_obj = self.validate_paths(state)
        self.logger.info("开始 MinerU 结构化解析 | file=%s", pdf_path_obj.name)

        #上传获取下载地址
        zip_url = self.upload_and_poll(pdf_path_obj)
        self.logger.info("MinerU 解析结果已就绪 | file=%s", pdf_path_obj.name)

        md_path = self.download_and_extract(zip_url,output_dir_obj,pdf_path_obj.stem)

        with open(md_path,"r",encoding="utf-8") as f:
            md_content = f.read()

        state["md_content"] = md_content
        state["md_path"] = md_path
        self.logger.info(
            "PDF 转 Markdown 完成 | file=%s | chars=%d | md_path=%s",
            pdf_path_obj.name,
            len(md_content),
            md_path,
        )
        return state


    def validate_paths(self, state: ImportGraphState):
        pdf_path = state.get("pdf_path")
        if not pdf_path:
            raise StateFieldError(field="pdf_path",expected_type = str)
        file_dir = state.get("file_dir")
        if not file_dir:
            raise StateFieldError(field="file_dir", expected_type=str)
        pdf_path_obj = Path(pdf_path)
        output_dir_obj =  Path(file_dir)

        if not pdf_path_obj.exists():
            raise FileProcessingError(message=f"输入文件不存在:{pdf_path}")
        if not output_dir_obj.exists():
            raise FileProcessingError(message=f"输出目录不存在:{output_dir_obj}")
        return pdf_path_obj,output_dir_obj

    def upload_and_poll(self, pdf_path_obj):
        #校验api_token 和base_url
        api_token = mineru_config.api_token
        base_url = mineru_config.base_url
        if not api_token:
            raise FileProcessingError(message="api_token未配置")
        if not base_url:
            raise FileProcessingError(message="base_url未配置")
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_token}"
        }
        data = {
            "files": [
                {"name":pdf_path_obj.name}
            ],
            "model_version": "vlm"
        }
        response = requests.post(f"{base_url}/file-urls/batch", headers=header, json=data)
        if response.status_code != 200:
            raise FileProcessingError(message=f"申请文件上传失败:{response.text}")
        result = response.json()
        if result["code"] != 0:
            raise FileProcessingError(message=f"申请文件上传失败:{result.get('message')}")

        batch_id = result["data"]["batch_id"]
        url = result["data"]["file_urls"][0]
        self.logger.info(
            "MinerU 上传地址申请成功 | batch_id=%s | file=%s",
            batch_id,
            pdf_path_obj.name,
        )
        #上传文件
        with open(pdf_path_obj, 'rb') as f:
            res_upload = requests.put(url, data=f)
            if res_upload.status_code == 200:
                self.logger.info(
                    "PDF 上传成功 | batch_id=%s | file=%s",
                    batch_id,
                    pdf_path_obj.name,
                )
            else:
                self.logger.error(
                    "PDF 上传失败 | batch_id=%s | file=%s | status=%s",
                    batch_id,
                    pdf_path_obj.name,
                    res_upload.status_code,
                )
        # 获取下载连接
        poll_url = f"{base_url}/extract-results/batch/{batch_id}"
        start_time =  time.time()
        timeout_seconds = 600
        poll_interval = 3
        last_progress_log = -15

        while True:
            end_time = time.time() - start_time
            if end_time > timeout_seconds:
                raise FileProcessingError(message="获得下载地址超时")
            try:
                res_poll = requests.get(url=poll_url, headers=header,timeout =10)
            except Exception as e:
                self.logger.warning(
                    "MinerU 状态轮询异常，将继续重试 | batch_id=%s | error=%s",
                    batch_id,
                    e,
                )
                time.sleep(poll_interval)
                continue
            if res_poll.status_code != 200:
                return PdfConversionError(f"http请求失败,状态码:{res_poll.status_code},响应内容:{res_poll}")

            poll_data = res_poll.json()
            if poll_data["code"] != 0:
                raise PdfConversionError(f"任务失败,错误信息:{poll_data.get('message')}")

            extract_results = poll_data["data"]["extract_result"]
            extract_result = extract_results[0]
            extract_state = extract_result["state"]

            if extract_state == "done":
                full_zip_url = extract_result["full_zip_url"]
                return full_zip_url
            elif extract_state == "failed":
                err_msg = extract_state.get("err_msg","未知错误，无具体信息")
                return PdfConversionError(f"任务解析失败:{err_msg}")
            else:
                elapsed_seconds = int(end_time)
                if elapsed_seconds - last_progress_log >= 15:
                    self.logger.info(
                        "MinerU 解析中 | batch_id=%s | state=%s | elapsed=%ds",
                        batch_id,
                        extract_state,
                        elapsed_seconds,
                    )
                    last_progress_log = elapsed_seconds
                time.sleep(poll_interval)


    def download_and_extract(self, zip_url, output_dir_obj, stem):
        self.logger.info("开始下载并解压 MinerU 结果 | file_title=%s", stem)
        response = requests.get(zip_url)
        if response.status_code != 200:
            raise FileProcessingError(message=f"获取下载文件失败:{response.text}")
        zip_save_path = output_dir_obj / f"{stem}.zip"
        with open(zip_save_path, 'wb') as f:
            f.write(response.content)
        extract_target_dir =  output_dir_obj / stem
        extract_target_dir.mkdir(parents=True,exist_ok=True)
        #解压
        with zipfile.ZipFile(zip_save_path, 'r') as zip_ref:
            zip_ref.extractall(extract_target_dir)

        target_md_file = extract_target_dir / "full.md"
        name = target_md_file.with_name(f"{stem}.md")
        target_md_file.rename(name)
        self.logger.info(
            "MinerU 结果解压完成 | file_title=%s | zip_bytes=%d",
            stem,
            len(response.content),
        )

        return str(name.absolute())


if "__main__" == __name__:
    setup_logging()

    state ={
        "pdf_path": r"",
        "file_dir": "E:/code/py/rag/output",
    }
    node =  NodePDFToMD()
    result = node(state)
    print(json.dumps(result,ensure_ascii=False,indent=4))
