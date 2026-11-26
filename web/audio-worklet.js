/**
 * Otto capture worklet.
 *
 * Runs on the audio render thread: takes whatever the microphone delivers,
 * downsamples it to 16 kHz mono (the wire format for browser -> server PCM),
 * converts to Int16LE and posts fixed-size chunks (~32 ms) to the main thread,
 * which owns the energy VAD and the websocket.
 *
 * A box-filter decimation is used (average of `ratio` input samples per output
 * sample). Ratio is derived from the real context rate, so the processor works
 * whether the AudioContext runs at 16 kHz, 44.1 kHz or 48 kHz.
 */
class OttoCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.targetRate = opts.targetRate || 16000;
    this.chunkSamples = opts.chunkSamples || 512; // 32 ms @ 16 kHz
    this.ratio = Math.max(1, sampleRate / this.targetRate);
    this.acc = 0;
    this.sum = 0;
    this.count = 0;
    this.stage = new Int16Array(this.chunkSamples);
    this.stageLen = 0;
    this.port.onmessage = (event) => {
      if (event.data === 'flush') this.flush();
    };
  }

  flush() {
    if (this.stageLen === 0) return;
    const out = this.stage.slice(0, this.stageLen);
    this.stageLen = 0;
    this.port.postMessage(out, [out.buffer]);
  }

  emit(value) {
    let sample;
    if (value > 1) sample = 32767;
    else if (value < -1) sample = -32768;
    else sample = (value * 32767) | 0;
    this.stage[this.stageLen++] = sample;
    if (this.stageLen >= this.chunkSamples) this.flush();
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channel = input[0];
    if (!channel) return true;

    for (let i = 0; i < channel.length; i++) {
      this.sum += channel[i];
      this.count += 1;
      this.acc += 1;
      if (this.acc >= this.ratio) {
        this.emit(this.sum / this.count);
        this.acc -= this.ratio;
        this.sum = 0;
        this.count = 0;
      }
    }
    return true;
  }
}

registerProcessor('otto-capture', OttoCaptureProcessor);
