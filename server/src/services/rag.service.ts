import path from "node:path";
import { fileURLToPath } from "node:url";
import { MongoClient } from "mongodb";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { TextLoader } from "@langchain/classic/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ---- MongoDB native client ----
let mongoClient: MongoClient | null = null;

const getMongoClient = async (): Promise<MongoClient> => {
  if (!mongoClient) {
    mongoClient = new MongoClient(process.env.MONGODB_URI || "");
    await mongoClient.connect();
  }
  return mongoClient;
};

// ---- Google GenAI Embeddings ----
// gemini-embedding-001 → default 3072 dimensions (FREE, same API key as Gemini chat)
const getEmbeddings = () => {
  if (!process.env.GOOGLE_API_KEY) {
    throw new Error("GOOGLE_API_KEY is not set in .env!");
  }
  return new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GOOGLE_API_KEY,
    model: "gemini-embedding-001",
  });
};

// ---- Vector Store ----
const getVectorStore = async () => {
  const client = await getMongoClient();
  const collection = client.db("edureach_db").collection("knowledge_docs");

  return new MongoDBAtlasVectorSearch(getEmbeddings(), {
    collection: collection as any,
    indexName: "edureachvectorindex",
    textKey: "text",
    embeddingKey: "embedding",
  });
};

// --- Initialize Knowledge Base ---
// A) INDEXING — runs ONCE at server startup
// ============================================
export const initializeKnowledgeBase = async (): Promise<void> => {
  const client = await getMongoClient();
  const collection = client.db("edureach_db").collection("knowledge_docs");

  // Check if docs exist WITH valid (non-empty) embeddings
  const docWithEmbedding = await collection.findOne({
    embedding: { $exists: true, $not: { $size: 0 } },
  });

  if (docWithEmbedding) {
    const count = await collection.countDocuments();
    console.log(` Knowledge base ready (${count} chunks with embeddings)`);
    return;
  }

  // If docs exist but embeddings are empty → delete and re-index
  const existingCount = await collection.countDocuments();
  if (existingCount > 0) {
    console.log(` Found ${existingCount} chunks with EMPTY embeddings — deleting & re-indexing...`);
    await collection.deleteMany({});
  }

  console.log(" Indexing knowledge base...");

  // Verify API key FIRST with a test embedding
  const embeddings = getEmbeddings();
  try {
    const testResult = await embeddings.embedQuery("test");
    console.log(` API key OK — embedding dimensions: ${testResult.length}`);
  } catch (error: any) {
    console.error(" Embedding test failed!");
    console.error("   Error:", error.message || error);
    console.error("   Get key from: https://aistudio.google.com/apikey");
    throw error;
  }

  // LOAD
  const filePath = path.join(__dirname, "../../knowledge-base/edureach-knowledge.txt");
  const loader = new TextLoader(filePath);
  const docs = await loader.load();
  if (docs.length === 0) {
    throw new Error("No documents found in knowledge base file");
  }
  const totalCharacters = docs.reduce((sum, doc) => sum + doc.pageContent.length, 0);
  console.log(`    Loaded ${totalCharacters} characters`);

  // SPLIT
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const allSplits = await splitter.splitDocuments(docs);
  console.log(`    Split into ${allSplits.length} chunks`);

  // EMBED + STORE
  const vectorStore = new MongoDBAtlasVectorSearch(embeddings, {
    collection: collection as any,
    indexName: "edureachvectorindex",
    textKey: "text",
    embeddingKey: "embedding",
  });

  await vectorStore.addDocuments(allSplits);

  // VERIFY
  const verifyDoc = await collection.findOne({
    embedding: { $exists: true, $not: { $size: 0 } },
  });

  if (verifyDoc && Array.isArray(verifyDoc.embedding) && verifyDoc.embedding.length > 0) {
    console.log(`    ${allSplits.length} chunks stored (${verifyDoc.embedding.length}D embeddings)`);
    console.log(`     IMPORTANT: Create Atlas Vector Search index with numDimensions: ${verifyDoc.embedding.length}`);
  } else {
    await collection.deleteMany({});
    throw new Error(" Embeddings are empty! Google API returned no vectors.");
  }
};

// --- Get RAG Response ---
export const getRAGResponse = async (question: string): Promise<string> => {
  try {
    const vectorStore = await getVectorStore();
    let retrievedDocs;
    try {
      retrievedDocs = await vectorStore.similaritySearch(question, 4);
      console.log(` RAG retrieved ${retrievedDocs.length} chunks for: "${question}"`);
    } catch (error) {
      console.error(" RAG retrieval failed. Check MongoDB Atlas connection and vector index config.");
      console.error("   Expected database: edureach_db");
      console.error("   Expected collection: knowledge_docs");
      console.error("   Expected vector index: edureachvectorindex");
      throw error;
    }

    if (retrievedDocs.length === 0) {
      return "I don't have that information right now. Click Talk to Us to speak with a counselor.";
    }

    const model = new ChatGoogleGenerativeAI({
      apiKey: process.env.GOOGLE_API_KEY || "",
      model: "gemini-3.5-flash-lite",
      temperature: 0.3,
    });

    const context = retrievedDocs
      .map((doc, index) => `Source ${index + 1}: ${doc.pageContent}`)
      .join("\n\n");

    let result;
    try {
      result = await model.invoke([
        new SystemMessage(
          "You are EduReach Bot, a helpful AI counselor for EduReach College, Hyderabad. " +
            "Answer only from the provided knowledge base context. " +
            "Be concise, friendly, and professional. " +
            "Always format responses cleanly: use bold category headings (e.g. **B.Tech Programs (4 years):**), bullet points for lists on new lines, and line breaks between sections. " +
            "If the context does not contain the answer, say exactly: " +
            "\"I don't have that information right now. Click Talk to Us to speak with a counselor.\""
        ),
        new HumanMessage(`Knowledge base context:\n${context}\n\nUser question: ${question}`),
      ]);
    } catch (error) {
      console.error(" RAG generation failed. Check GOOGLE_API_KEY and Gemini model access.");
      throw error;
    }

    if (!result.content) {
      return "I couldn't generate a response. Please try again.";
    }

    return typeof result.content === "string"
      ? result.content
      : JSON.stringify(result.content);
  } catch (error) {
    console.error(" RAG Agent Error:", error);
    return "I'm having trouble right now. Please try again or click 'Talk to Us'.";
  }
};
