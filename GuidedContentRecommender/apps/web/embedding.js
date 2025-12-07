// https://huggingface.co/optimum/all-MiniLM-L6-v2/tree/main
import * as ort from "onnxruntime-node";
import { AutoTokenizer } from "@huggingface/transformers";
// import { AutoTokenizer } from "@xenova/transformers";

async function loadSession(model_uri) {
  let session = await ort.InferenceSession.create(model_uri);
  return session;
}

async function loadTokenizer(model_dir) {
  let tokenizer = await AutoTokenizer.from_pretrained(model_dir, {
    local_files_only: true,
  });
  return tokenizer;
}

async function generateEmbedding(session, tokenizer, text) {
  if (!session || !tokenizer) {
    throw new Error("Model and Tokenizer not loaded. Please check!");
  }

  // Tokenize
  const encoded = await tokenizer(text, {
    padding: true,
    truncation: true,
    max_length: 128,
  });

  console.log(typeof encoded.input_ids);
  console.log("---");
  console.log(encoded.attention_mask.ort_tensor.cpuData);
  console.log("---");
  console.log(encoded.token_type_ids);
  console.log("---");

  const dataAttentionMask = new Float32Array(
    Array.from(encoded.attention_mask.ort_tensor.cpuData).map((bigint) =>
      Number(bigint)
    )
  );
  const tensorAttentionMask = new ort.Tensor(
    "float32",
    dataAttentionMask,
    [1, 11] // TODO change dimension
  );
  console.log(tensorAttentionMask);

  // const t = new ort.Tensor("int64", [1, 2, 4], [1, 3]);
  // console.log(t);
  // console.log(typeof t);
  //
  //

  // console.log(session.inputNames);
  // console.log(session.outputNames);

  const feeds = {
    input_ids: encoded.input_ids.ort_tensor,
    attention_mask: encoded.attention_mask.ort_tensor,
    token_type_ids: encoded.token_type_ids.ort_tensor,
  };

  // const feeds = {
  //   input_ids: encoded.input_ids.ort_tensor,
  //   attention_mask: encoded.attention_mask.ort_tensor,
  //   token_type_ids: encoded.token_type_ids.ort_tensor,
  // };

  // Run inference
  const results = await session.run(feeds);
  console.log(results);

  // return results;
}

async function main() {
  const model_dir = "./model";
  const model_path = model_dir + "/model.onnx";

  let session = await loadSession(model_path);
  let tokenizer = await loadTokenizer(model_dir);

  let sentence = "I am sleepy and I would like to relax";
  let output = generateEmbedding(session, tokenizer, sentence);
  console.log(output);
}

main();
