import fs from 'fs';
import path from 'path';
import { pathToFileURL } from 'url';

const DEFAULT_INPUT_DIR =
  '/mnt/shared-storage-user/wenxiaoyu/hezhida/TROJail/data/ultrachat_200k/data';
const DEFAULT_OUTPUT_DIR =
  '/mnt/shared-storage-user/wenxiaoyu/hezhida/TROJail/matric-rollout/DPO_balance/Helpful';
const DEFAULT_HYPARQUET_MODULE =
  '/mnt/shared-storage-user/wenxiaoyu/hezhida/.tmp_npm_parquet/node_modules/hyparquet/src/node.js';
const DEFAULT_SYSTEM_PROMPT =
  'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.';

function parseArgs(argv) {
  const args = {
    inputDir: DEFAULT_INPUT_DIR,
    outputDir: DEFAULT_OUTPUT_DIR,
    limit: 9000,
    chunkSize: 1000,
    batchSize: 256,
    cleanOutput: false,
    hyparquetModule: DEFAULT_HYPARQUET_MODULE,
    systemPrompt: DEFAULT_SYSTEM_PROMPT,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--input-dir') {
      args.inputDir = argv[++i];
    } else if (arg === '--output-dir') {
      args.outputDir = argv[++i];
    } else if (arg === '--limit') {
      args.limit = Number(argv[++i]);
    } else if (arg === '--chunk-size') {
      args.chunkSize = Number(argv[++i]);
    } else if (arg === '--batch-size') {
      args.batchSize = Number(argv[++i]);
    } else if (arg === '--hyparquet-module') {
      args.hyparquetModule = argv[++i];
    } else if (arg === '--system-prompt') {
      args.systemPrompt = argv[++i];
    } else if (arg === '--clean-output') {
      args.cleanOutput = true;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  if (!Number.isInteger(args.limit) || args.limit <= 0) {
    throw new Error('--limit must be a positive integer');
  }
  if (!Number.isInteger(args.chunkSize) || args.chunkSize <= 0) {
    throw new Error('--chunk-size must be a positive integer');
  }
  if (!Number.isInteger(args.batchSize) || args.batchSize <= 0) {
    throw new Error('--batch-size must be a positive integer');
  }

  return args;
}

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function cleanOutputDir(outputDir) {
  ensureDir(outputDir);
  for (const name of fs.readdirSync(outputDir)) {
    if (name.endsWith('.json') || name.endsWith('.jsonl')) {
      fs.unlinkSync(path.join(outputDir, name));
    }
  }
}

function listTrainFiles(inputDir) {
  return fs
    .readdirSync(inputDir)
    .filter((name) => /^train_sft-\d+-of-\d+.*\.parquet$/.test(name))
    .sort((a, b) => {
      const aNum = Number(a.match(/^train_sft-(\d+)-of-/)[1]);
      const bNum = Number(b.match(/^train_sft-(\d+)-of-/)[1]);
      return aNum - bNum;
    })
    .map((name) => path.join(inputDir, name));
}

function extractFirstPair(rawRow) {
  const messages = Array.isArray(rawRow.messages) ? rawRow.messages : null;
  let userContent = typeof rawRow.prompt === 'string' ? rawRow.prompt.trim() : '';
  let response = '';

  if (messages && messages.length >= 2) {
    const firstUser = messages.find(
      (message) => message && message.role === 'user' && typeof message.content === 'string' && message.content.trim(),
    );
    const firstAssistant = messages.find(
      (message) =>
        message && message.role === 'assistant' && typeof message.content === 'string' && message.content.trim(),
    );

    if (!userContent && firstUser) {
      userContent = firstUser.content.trim();
    }
    if (firstAssistant) {
      response = firstAssistant.content.trim();
    }
  }

  if (!userContent || !response) {
    return null;
  }

  return {
    prompt: [
      { role: 'system', content: DEFAULT_SYSTEM_PROMPT },
      { role: 'user', content: userContent },
    ],
    response,
  };
}

function flushShard(outputDir, shardIndex, rows) {
  const filename = `helpful_part_${String(shardIndex).padStart(5, '0')}.jsonl`;
  const fullPath = path.join(outputDir, filename);
  const payload = rows.map((row) => JSON.stringify(row)).join('\n') + '\n';
  fs.writeFileSync(fullPath, payload, 'utf-8');
  return { file: filename, numRows: rows.length };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const hyparquetModule = await import(pathToFileURL(args.hyparquetModule).href);
  const { asyncBufferFromFile, parquetMetadataAsync, parquetReadObjects } = hyparquetModule;

  const inputFiles = listTrainFiles(args.inputDir);
  if (inputFiles.length === 0) {
    throw new Error(`No train_sft parquet files found under ${args.inputDir}`);
  }

  if (args.cleanOutput) {
    cleanOutputDir(args.outputDir);
  } else {
    ensureDir(args.outputDir);
  }

  const summary = {
    inputDir: args.inputDir,
    inputFiles,
    outputDir: args.outputDir,
    limit: args.limit,
    chunkSize: args.chunkSize,
    batchSize: args.batchSize,
    systemPrompt: args.systemPrompt,
    selectedRows: 0,
    skippedRows: 0,
    filesUsed: [],
    outputFiles: [],
  };

  let shardRows = [];
  let shardIndex = 1;

  for (const inputFile of inputFiles) {
    if (summary.selectedRows >= args.limit) {
      break;
    }

    const file = await asyncBufferFromFile(inputFile);
    const metadata = await parquetMetadataAsync(file);
    const numRows = Number(metadata.num_rows);

    const fileSummary = {
      file: inputFile,
      totalRows: numRows,
      selectedRows: 0,
      skippedRows: 0,
    };

    for (let rowStart = 0; rowStart < numRows && summary.selectedRows < args.limit; rowStart += args.batchSize) {
      const rowEnd = Math.min(numRows, rowStart + args.batchSize);
      const rows = await parquetReadObjects({ file, rowStart, rowEnd });

      for (const rawRow of rows) {
        if (summary.selectedRows >= args.limit) {
          break;
        }

        const normalized = extractFirstPair(rawRow);
        if (!normalized) {
          summary.skippedRows += 1;
          fileSummary.skippedRows += 1;
          continue;
        }

        normalized.prompt[0].content = args.systemPrompt;
        shardRows.push(normalized);
        summary.selectedRows += 1;
        fileSummary.selectedRows += 1;

        if (shardRows.length >= args.chunkSize) {
          summary.outputFiles.push(flushShard(args.outputDir, shardIndex, shardRows));
          shardRows = [];
          shardIndex += 1;
        }
      }
    }

    summary.filesUsed.push(fileSummary);
  }

  if (shardRows.length > 0) {
    summary.outputFiles.push(flushShard(args.outputDir, shardIndex, shardRows));
  }

  const summaryPath = path.join(args.outputDir, 'summary.json');
  fs.writeFileSync(summaryPath, `${JSON.stringify(summary, null, 2)}\n`, 'utf-8');
  process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
