import json
import logging
import time
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
        logging.info(f"{self.name}节点开始执行...")

        #参数检查
        pdf_path_obj,output_dir_obj = self.validate_paths(state)

        #上传获取下载地址
        zip_url = self.upload_and_poll(pdf_path_obj)
        logging.info(f"下载地址：{zip_url}")

        md_path = self.download_and_extract(zip_url,output_dir_obj,pdf_path_obj.stem)

        with open(md_path,"r",encoding="utf-8") as f:
            md_content = f.read()

        state["md_content"] = md_content
        state["md_path"] = md_path
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
        logging.info('batch_id:{},url:{}'.format(batch_id, url))
        #上传文件
        with open(pdf_path_obj, 'rb') as f:
            res_upload = requests.put(url, data=f)
            if res_upload.status_code == 200:
                logging.info(f"{url} upload success")
            else:
                logging.info(f"{url} upload failed")
        # 获取下载连接
        poll_url = f"{base_url}/extract-results/batch/{batch_id}"
        start_time =  time.time()
        timeout_seconds = 600
        poll_interval = 3

        while True:
            end_time = time.time() - start_time
            if end_time > timeout_seconds:
                raise FileProcessingError(message="获得下载地址超时")
            try:
                res_poll = requests.get(url=poll_url, headers=header,timeout =10)
            except Exception as e:
                self.logger.error(f"轮询接口异常：{e}")
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
                self.logger.info(f"任务轮询中...已耗时{int(end_time)}s,状态:{extract_state}")
                time.sleep(poll_interval)





    def download_and_extract(self, zip_url, output_dir_obj, stem):
        pass

if "__main__" == __name__:
    setup_logging()

    state ={
        "pdf_path": r"E:\code\py\掌柜智库课件0525\掌柜智库课件0525\2.资料\04-设备手册汇总\doc\hak180产品安全手册.pdf",
        "file_dir": "E:/code/py/rag/output",
    }
    node =  NodePDFToMD()
    result = node(state)
    print(json.dumps(result,ensure_ascii=False,indent=4))