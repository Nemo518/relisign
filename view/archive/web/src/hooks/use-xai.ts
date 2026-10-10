import { useMutation } from "@tanstack/react-query";
import { api } from "@/api/client";
import type { XAIRequest, XAIResponse } from "@/types/api";

export function useXai() {
  return useMutation<XAIResponse, Error, XAIRequest>({
    mutationFn: (req) => api.xai(req),
  });
}
