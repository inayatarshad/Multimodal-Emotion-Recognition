import { useCallback, useEffect, useRef, useState } from 'react';
import { api, websocketUrl } from '../api/client';
import type { CompareResponse, CorruptionRequest } from '../api/types';

type Status = 'connecting' | 'live' | 'polling' | 'error';

/**
 * Streams comparison results as the sliders move.
 *
 * Uses the WebSocket when it is available and silently falls back to POST /api/compare
 * when it is not — a demo that breaks behind a proxy that drops WebSocket upgrades is
 * worse than one that is slightly less smooth.
 *
 * Requests are coalesced: dragging a slider fires many changes, and only the most recent
 * one matters. Anything sent while a response is outstanding replaces the pending
 * request rather than queueing behind it, so the readout tracks the thumb instead of
 * lagging further and further behind it.
 */
export function useLiveCompare(
  sampleId: string | null,
  corruption: CorruptionRequest,
  models: string[],
) {
  const [data, setData] = useState<CompareResponse | null>(null);
  const [status, setStatus] = useState<Status>('connecting');
  const [busy, setBusy] = useState(false);

  const socketRef = useRef<WebSocket | null>(null);
  const inFlightRef = useRef(false);
  const pendingRef = useRef<{ corruption: CorruptionRequest; models: string[] } | null>(null);

  useEffect(() => {
    let cancelled = false;
    let socket: WebSocket | null = null;

    try {
      socket = new WebSocket(websocketUrl('/ws/live'));
    } catch {
      setStatus('polling');
      return;
    }

    socket.onopen = () => {
      if (!cancelled) setStatus('live');
    };
    socket.onmessage = (event) => {
      if (cancelled) return;
      try {
        const message = JSON.parse(event.data as string) as CompareResponse & { error?: string };
        if (!message.error) setData(message);
      } catch {
        /* a malformed frame is not worth tearing the connection down for */
      }
      inFlightRef.current = false;
      setBusy(false);
      flush();
    };
    socket.onerror = () => {
      if (!cancelled) setStatus('polling');
    };
    socket.onclose = () => {
      if (!cancelled) setStatus((current) => (current === 'live' ? 'polling' : current));
    };
    socketRef.current = socket;

    return () => {
      cancelled = true;
      socketRef.current = null;
      // Closing a CONNECTING socket logs "closed before the connection is established".
      // React 18 StrictMode mounts effects twice in dev, so this fires on every reload
      // unless the close is deferred until the handshake finishes.
      if (socket.readyState === WebSocket.CONNECTING) {
        socket.onopen = () => socket.close();
      } else {
        socket.close();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const send = useCallback(
    async (payload: { corruption: CorruptionRequest; models: string[] }) => {
      if (!sampleId) return;
      const socket = socketRef.current;
      inFlightRef.current = true;
      setBusy(true);

      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(
          JSON.stringify({
            sample_id: sampleId,
            models: payload.models,
            corruption: payload.corruption,
          }),
        );
        return;
      }

      try {
        const response = await api.compare({
          sample_id: sampleId,
          models: payload.models,
          corruption: payload.corruption,
        });
        setData(response);
        // Must not clobber 'live'. The very first request races the WebSocket handshake:
        // the socket is still CONNECTING, so this REST fallback runs, and it used to
        // resolve *after* onopen had set 'live' — leaving the indicator permanently
        // claiming to poll while every later request in fact went over the socket.
        setStatus((current) => (current === 'live' ? current : 'polling'));
      } catch {
        setStatus('error');
      } finally {
        inFlightRef.current = false;
        setBusy(false);
        flush();
      }
    },
    [sampleId],
  );

  const flush = useCallback(() => {
    const pending = pendingRef.current;
    if (pending && !inFlightRef.current) {
      pendingRef.current = null;
      void send(pending);
    }
  }, [send]);

  useEffect(() => {
    if (!sampleId || models.length === 0) return;
    const payload = { corruption, models };
    if (inFlightRef.current) {
      pendingRef.current = payload; // coalesce: keep only the newest
      return;
    }
    void send(payload);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sampleId, JSON.stringify(corruption), models.join(','), send]);

  return { data, status, busy };
}
