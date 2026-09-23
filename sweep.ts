import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { createWriteStream } from "node:fs";
import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { config as loadEnvironment } from "dotenv";

import {
  isJudgeId,
  JUDGES,
  JUDGE_IDS,
  type Judge,
  type JudgeId,
} from "./src/judges.js";

loadEnvironment({ quiet: true });

const PROJECT_DIRECTORY = fileURLToPath(new URL(".", import.meta.url));
const URL_PATTERN = /https?:\/\/[^"\s]+/gu;

type Provider = Judge["provider"];

type CommandResult = {
  exitCode: number;
};

function printUsage(): void {
  process.stdout.write(`Usage: pnpm sweep -- [judge ...]

Run all judges, or pass one or more judge IDs:
  ${JUDGE_IDS.join("\n  ")}

Configuration is read from the process environment and .env.
`);
}

function parseSelectedJudges(args: string[]): JudgeId[] {
  if (args.length === 0) {
    return JUDGE_IDS;
  }
  const unknownJudges = args.filter((value) => !isJudgeId(value));
  if (unknownJudges.length > 0) {
    throw new Error(
      `Unknown judge${unknownJudges.length === 1 ? "" : "s"}: ${unknownJudges.join(
        ", "
      )}. Choose from: ${JUDGE_IDS.join(", ")}`
    );
  }
  return args as JudgeId[];
}

function getSweepRunId({
  configured = process.env.SWEEP_RUN_ID,
  now = new Date(),
  uuid = randomUUID(),
}: {
  /** Explicit run ID, usually from SWEEP_RUN_ID. */
  configured?: string;
  /** Timestamp used when an ID is generated. */
  now?: Date;
  /** Unique suffix used when an ID is generated. */
  uuid?: string;
} = {}): string {
  if (configured) {
    return configured;
  }
  const timestamp = now
    .toISOString()
    .replaceAll(/[-:]/gu, "")
    .replace(/\.\d{3}Z$/u, "Z");
  return `${timestamp}-${uuid}`;
}

function validateApiKeys({ judges }: { judges: JudgeId[] }): void {
  const requiredVariables = new Set(
    judges.map((judgeId) => JUDGES[judgeId].apiKeyEnvironmentVariable)
  );
  const missingVariables = [...requiredVariables].filter(
    (variable) => !process.env[variable]
  );
  if (missingVariables.length > 0) {
    throw new Error(
      `Missing API key${missingVariables.length === 1 ? "" : "s"}: ${missingVariables.join(
        ", "
      )}. Copy .env.example to .env and set the keys required by the selected judges.`
    );
  }
}

function groupJudgesByProvider({
  judges,
}: {
  judges: JudgeId[];
}): Map<Provider, JudgeId[]> {
  const lanes = new Map<Provider, JudgeId[]>();
  for (const judgeId of judges) {
    const provider = JUDGES[judgeId].provider;
    lanes.set(provider, [...(lanes.get(provider) ?? []), judgeId]);
  }
  return lanes;
}

function runCommand({
  args,
  environment = process.env,
  logFile,
}: {
  /** Arguments passed to pnpm. */
  args: string[];
  /** Complete child-process environment. */
  environment?: NodeJS.ProcessEnv;
  /** Optional file that receives stdout and stderr. */
  logFile?: string;
}): Promise<CommandResult> {
  return new Promise((resolve, reject) => {
    const log = logFile === undefined ? undefined : createWriteStream(logFile);
    const child = spawn("pnpm", args, {
      cwd: PROJECT_DIRECTORY,
      env: environment,
      stdio: log === undefined ? "inherit" : ["ignore", "pipe", "pipe"],
    });

    if (log !== undefined) {
      child.stdout?.pipe(log, { end: false });
      child.stderr?.pipe(log, { end: false });
    }

    child.on("error", (error) => {
      log?.end();
      reject(error);
    });
    child.on("close", (exitCode) => {
      log?.end();
      resolve({ exitCode: exitCode ?? 1 });
    });
  });
}

async function runJudge({
  judgeId,
  sweepRunId,
  logDirectory,
  reportDirectory,
}: {
  /** Judge configuration to execute. */
  judgeId: JudgeId;
  /** Run identifier written to every Phoenix experiment. */
  sweepRunId: string;
  /** Directory for complete Vitest output. */
  logDirectory: string;
  /** Directory for Phoenix reporter artifacts. */
  reportDirectory: string;
}): Promise<JudgeId | undefined> {
  const logFile = join(logDirectory, `${judgeId}.log`);
  const judgeReportDirectory = join(reportDirectory, judgeId);
  await mkdir(judgeReportDirectory, { recursive: true });
  process.stdout.write(
    `${new Date().toISOString()} start ${judgeId} (log: ${logFile})\n`
  );
  const { exitCode } = await runCommand({
    args: ["run", "evals"],
    environment: {
      ...process.env,
      JUDGE: judgeId,
      PHOENIX_TEST_REPORT_DIR: judgeReportDirectory,
      SWEEP_RUN_ID: sweepRunId,
    },
    logFile,
  });
  process.stdout.write(
    `${new Date().toISOString()} done ${judgeId} (exit ${exitCode})\n`
  );
  return exitCode === 0 ? undefined : judgeId;
}

async function runProviderLane({
  judges,
  sweepRunId,
  logDirectory,
  reportDirectory,
}: {
  /** Judges that share one provider rate limit. */
  judges: JudgeId[];
  /** Run identifier written to every Phoenix experiment. */
  sweepRunId: string;
  /** Directory for complete Vitest output. */
  logDirectory: string;
  /** Directory for Phoenix reporter artifacts. */
  reportDirectory: string;
}): Promise<JudgeId[]> {
  const failedJudges: JudgeId[] = [];
  for (const judgeId of judges) {
    const failedJudge = await runJudge({
      judgeId,
      sweepRunId,
      logDirectory,
      reportDirectory,
    });
    if (failedJudge !== undefined) {
      failedJudges.push(failedJudge);
    }
  }
  return failedJudges;
}

async function collectUrls(directory: string): Promise<string[]> {
  const entries = await readdir(directory, { withFileTypes: true });
  const urls = new Set<string>();
  for (const entry of entries) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      for (const url of await collectUrls(path)) {
        urls.add(url);
      }
      continue;
    }
    const contents = await readFile(path, "utf8");
    for (const match of contents.matchAll(URL_PATTERN)) {
      urls.add(match[0]);
    }
  }
  return [...urls].sort();
}

async function main(): Promise<void> {
  const args = process.argv.slice(2).filter((value) => value !== "--");
  if (args.includes("--help") || args.includes("-h")) {
    printUsage();
    return;
  }

  const judges = parseSelectedJudges(args);
  validateApiKeys({ judges });
  const sweepRunId = getSweepRunId();
  const resultsDirectory = join(
    PROJECT_DIRECTORY,
    "analysis",
    "results",
    sweepRunId
  );
  const logDirectory = join(resultsDirectory, "logs");
  const reportDirectory = join(resultsDirectory, "reports");
  const failedJudgesFile = join(resultsDirectory, "failed-judges.txt");
  await mkdir(logDirectory, { recursive: true });
  await mkdir(reportDirectory, { recursive: true });

  process.stdout.write(`Sweep run ID: ${sweepRunId}\n`);
  process.stdout.write(`Results: ${resultsDirectory}\n`);

  const pricingSetup = await runCommand({ args: ["run", "setup:costs"] });
  if (pricingSetup.exitCode !== 0) {
    throw new Error("Phoenix pricing setup failed; no judge calls were made.");
  }

  const lanes = groupJudgesByProvider({ judges });
  const failuresByLane = await Promise.all(
    [...lanes.values()].map((laneJudges) =>
      runProviderLane({
        judges: laneJudges,
        sweepRunId,
        logDirectory,
        reportDirectory,
      })
    )
  );
  const failedJudges = failuresByLane.flat();
  await writeFile(
    failedJudgesFile,
    failedJudges.length === 0 ? "" : `${failedJudges.join("\n")}\n`
  );

  const experimentUrls = await collectUrls(reportDirectory);
  if (experimentUrls.length > 0) {
    process.stdout.write(`Experiment links\n${experimentUrls.join("\n")}\n`);
  }
  if (failedJudges.length > 0) {
    process.stdout.write(
      `Judges with non-zero exits: ${failedJudges.join(
        ", "
      )}. Acceptance-gate failures are retained as benchmark data; inspect the logs before resuming.\n`
    );
  }
}

try {
  await main();
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(`Error: ${message}\n`);
  process.exitCode = 1;
}
