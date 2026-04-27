import { useMutation } from "@tanstack/react-query";
import { predictXray, PredictResponse } from "@/lib/api";

export function usePredict() {
  return useMutation<PredictResponse, Error, File>({
    mutationFn: predictXray,
  });
}
