import logging
from pathlib import Path

from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError, FileProcessingError
from processor.import_processor.state import ImportGraphState


class NodePDFToMD(BaseNode):
    """
    PDF 转 Markdown 节点：PDF结构化解析
    """

    name = "node_pdf_to_md"

    def process(self, state: ImportGraphState):
        logging.info(f"{self.name}节点开始执行...")

        pdf_path_obj,output_dir_obj = self.validate_paths(state)

        zip_url = self.upload_and_poll(pdf_path_obj)

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
        pass

    def download_and_extract(self, zip_url, output_dir_obj, stem):
        pass