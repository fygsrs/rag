import json
import logging

from langgraph.graph import StateGraph,END,START

from processor.import_processor.base import setup_logging
from processor.import_processor.nodes.a_node_entry import NodeEntry
from processor.import_processor.nodes.b_node_pdf_to_md import NodePDFToMD
from processor.import_processor.nodes.c_node_md_img import NodeMDImg
from processor.import_processor.nodes.d_node_document_split import NodeDocumentSplit
from processor.import_processor.nodes.e_node_item_name_recognition import NodeItemNameRecognition
from processor.import_processor.nodes.f_node_bge_embedding import NodeBGEEmbedding
from processor.import_processor.nodes.g_node_import_milvus import NodeImportMilvus
from processor.import_processor.state import ImportGraphState


class ImportWorkflow:
    def __init__(self,config = None):
        self._compiled_graph = None

    @staticmethod
    def route_after_entry(state: ImportGraphState):
        if state.get("is_pdf_read_enabled"):
            return "b_node_pdf_to_md"
        elif state.get("is_md_read_enabled"):
            return "c_node_md_img"
        else:
            return END

    @property
    def graph(self):
        logging.info("获取图实例")
        if self._compiled_graph is None:
            self._compiled_graph = self.build_graph()
        return self._compiled_graph

    def build_graph(self):
        """
        创建主图
        :return:
        """
        graph = StateGraph(ImportGraphState)
        graph.add_node("a_node_entry",NodeEntry())
        graph.add_node("b_node_pdf_to_md",NodePDFToMD())
        graph.add_node("c_node_md_img",NodeMDImg())
        graph.add_node("d_node_document_split",NodeDocumentSplit())
        graph.add_node("e_node_item_name_recognition",NodeItemNameRecognition())
        graph.add_node("f_node_bge_embedding",NodeBGEEmbedding())
        graph.add_node("g_node_import_milvus",NodeImportMilvus())
        graph.set_entry_point("a_node_entry")
        graph.add_conditional_edges(
            "a_node_entry",
             self.route_after_entry,
            {
                "c_node_md_img":"c_node_md_img",
                "b_node_pdf_to_md":"b_node_pdf_to_md",
                END:END
            }
        )
        graph.add_edge("b_node_pdf_to_md","c_node_md_img")
        graph.add_edge("c_node_md_img","d_node_document_split")
        graph.add_edge("d_node_document_split","e_node_item_name_recognition")
        graph.add_edge("e_node_item_name_recognition","f_node_bge_embedding")
        graph.add_edge("f_node_bge_embedding","g_node_import_milvus")
        graph.add_edge("g_node_import_milvus", END)
        graph_compile = graph.compile()
        return graph_compile

    def run(self,state:ImportGraphState,stream:bool = False):
        if stream:
            return self.graph.stream(state,stream_mode="values")
        else:
            return self.graph.invoke(state)

if __name__ == "__main__":
    setup_logging()
    workflow = ImportWorkflow()
    state ={
        "import_file_path":"E:/4/经历.md"
    }

    for event in workflow.run(state,stream = True):
        print(f"state:{event}")

    # final_state = workflow.run(state,stream = False)
    # print(json.dumps(final_state,ensure_ascii=False,indent=4))

