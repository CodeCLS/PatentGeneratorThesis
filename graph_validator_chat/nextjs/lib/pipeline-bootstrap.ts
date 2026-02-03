export interface PipelinePayload {
  text?: string;
  filename?: string;
  patent_id?: string;
  pdf_base64?: string;
}

export interface PipelineBootstrapResult {
  initialized: boolean;
  bootstrapped: boolean;
  error?: string;
}

export const PIPELINE_PAYLOAD_KEY = 'pipeline_payload';

let bootstrapPromise: Promise<PipelineBootstrapResult> | null = null;

const readPayload = (): PipelinePayload | null => {
  if (typeof window === 'undefined') return null;
  const stored = localStorage.getItem(PIPELINE_PAYLOAD_KEY);
  if (!stored) return null;
  try {
    return JSON.parse(stored) as PipelinePayload;
  } catch {
    return null;
  }
};

export const savePipelinePayload = (payload: PipelinePayload) => {
  if (typeof window === 'undefined') return;
  localStorage.setItem(PIPELINE_PAYLOAD_KEY, JSON.stringify(payload));
};

export const bootstrapPipelineIfNeeded = async (): Promise<PipelineBootstrapResult> => {
  if (bootstrapPromise) return bootstrapPromise;

  bootstrapPromise = (async () => {
    try {
      const statusRes = await fetch('/api/status');
      if (statusRes.ok) {
        const statusData = await statusRes.json();
        if (statusData.initialized) {
          return { initialized: true, bootstrapped: false };
        }
      }
    } catch {
      // Ignore status errors and try bootstrapping from payload.
    }

    try {
      const restoreRes = await fetch('/api/pipeline/restore', { method: 'POST' });
      if (restoreRes.ok) {
        const restoreData = await restoreRes.json().catch(() => ({}));
        if (restoreData?.success) {
          return { initialized: true, bootstrapped: true };
        }
      }
    } catch {
      // Ignore restore errors and try bootstrapping from payload.
    }

    const payload = readPayload();
    if (!payload) {
      return {
        initialized: false,
        bootstrapped: false,
        error: 'No saved pipeline payload found.',
      };
    }

    try {
      const response = await fetch('/api/pipeline/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || data.error) {
        throw new Error(data.error || 'Failed to initialize pipeline.');
      }
      return { initialized: true, bootstrapped: true };
    } catch (error) {
      return {
        initialized: false,
        bootstrapped: false,
        error: error instanceof Error ? error.message : 'Failed to initialize pipeline.',
      };
    }
  })();

  try {
    return await bootstrapPromise;
  } finally {
    bootstrapPromise = null;
  }
};
