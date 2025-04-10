from dotenv import load_dotenv
import pickle

load_dotenv()

DOCS_PATH = "docs/"


# Load and split the documents
# loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)


# pdf_file = os.path.join(DOCS_PATH, "KE_5.pdf")
# loader = MathpixPDFLoader(
#     pdf_file,
#     processed_file_format="md",
#     mathpix_api_id=os.environ.get("MATHPIX_API_ID"),
#     mathpix_api_key=os.environ.get("MATHPIX_API_KEY"),
# )
# docs = loader.load()

docs = pickle.load(open("docs/cached_KE_5.pkl", "rb"))
chunks = pickle.load(open("docs/KE_5_chunks.pkl", "rb"))
