import { useState } from "react";
import { useSpeech } from "@/hooks/use-speech";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Volume2, StopCircle } from "lucide-react";

interface Props {
  reference: string;
  prediction: string;
}

export function AudioConsole({ reference, prediction }: Props) {
  const { speak, stop } = useSpeech();
  const [rate, setRate] = useState(1.0);
  const [autoPlay, setAutoPlay] = useState(true);

  return (
    <div className="rounded-2xl border border-border bg-white p-4 dark:bg-slate-800">
      <div className="mb-3 text-sm font-semibold text-foreground">
        语音播报
      </div>
      <div className="mb-3 text-xs text-muted-foreground">
        支持直接播报当前预测文本，也可以切换为参考文本做现场对照。
      </div>

      <Slider
        value={rate}
        min={0.7}
        max={1.3}
        step={0.1}
        label="播报语速"
        onChange={setRate}
      />

      <div className="my-3">
        <Switch
          checked={autoPlay}
          onCheckedChange={setAutoPlay}
          label="推理完成后自动播报预测文本"
        />
      </div>

      <div className="flex gap-2">
        <Button
          size="sm"
          className="flex-1"
          onClick={() => speak(prediction, rate)}
          disabled={!prediction}
        >
          <Volume2 size={14} />
          播报预测文本
        </Button>
        <Button
          size="sm"
          variant="outline"
          className="flex-1"
          onClick={() => speak(reference, rate)}
          disabled={!reference}
        >
          <Volume2 size={14} />
          播报参考文本
        </Button>
        <Button size="sm" variant="ghost" onClick={stop}>
          <StopCircle size={14} />
          停止
        </Button>
      </div>
    </div>
  );
}
