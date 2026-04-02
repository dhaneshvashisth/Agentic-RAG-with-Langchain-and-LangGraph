import re
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

COLLECTION_NAME = "youtube_test_collection"


def extract_video_id(url: str):

    match = re.search(r"v=([^&]+)", url)

    if match:
        return match.group(1)

    raise ValueError("Invalid YouTube URL")


def fetch_transcript(video_id):

    youtube_transcript = YouTubeTranscriptApi()
    try:
        transcript = youtube_transcript.fetch(video_id)

    except Exception:

        transcript_list = youtube_transcript.list(video_id)

        transcript = transcript_list.find_generated_transcript(['en']).fetch()

    text = " ".join([t.text for t in transcript])

    return text


def ingest_video(url):

    video_id = extract_video_id(url)

    print("\nFetching transcript...")

    transcript = fetch_transcript(video_id)

    print("Transcript length:", len(transcript))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents(
        [transcript],
        metadatas=[{"video_id": video_id, "source_url": url}] 
    )

    print("Chunks created:", len(docs))

    embeddings = OpenAIEmbeddings()

    client = QdrantClient("localhost", port=6333)

    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        print("Collection created.")

    vectordb = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

    vectordb.add_documents(docs)

    print("\nEmbeddings stored in Qdrant")



if __name__ == "__main__":

    while True:

        url = input("\nEnter a youtube url (or type 'exit'): ")

        if url.strip().lower() in ["exit", "quit", "bye"]:
            print("Goodbye !")
            break

        ingest_video(url)