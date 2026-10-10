import { useMutation } from "@tanstack/react-query";
import { api } from "@/api/client";
import type { PredictRequest, PredictResponse } from "@/types/api";

export function usePrediction() {
  return useMutation<PredictResponse, Error, PredictRequest>({
    mutationFn: (req) => api.predict(req),
  });
}
