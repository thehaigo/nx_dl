# NxDl

Nx,exlaを使用して以下の書籍3章ニューラルネットワークを実装したサンプルコードです
https://www.oreilly.co.jp/books/9784873117584/

https://qiita.com/the_haigo/items/1a2f0b371a3644960251

ch3

```
$ iex -S mix
iex(1)> Dataset.download(:mnist)
iex(2)> Ch3.NeruralnetMnist.acc
Load Weight
Operating System: macOS
CPU Information: Intel(R) Core(TM) i7-7660U CPU @ 2.50GHz
Number of Available Cores: 4
Available memory: 16 GB
Elixir 1.11.3
Erlang 23.2.5

Benchmark suite executing with the following configuration:
warmup: 2 s
time: 5 s
memory time: 0 ns
parallel: 1
inputs: none specified
Estimated total run time: 14 s

Benchmarking batch...

21:43:15.655 [info]  XLA service 0x7f9050b21670 initialized for platform Host (this does not guarantee that XLA will be used). Devices:

21:43:15.664 [info]    StreamExecutor device (0): Host, Default Version
Benchmarking nomal...

Name            ips        average  deviation         median         99th %
batch          1.21         0.83 s    ±11.39%         0.83 s         1.00 s
nomal         0.149         6.71 s     ±0.00%         6.71 s         6.71 s

Comparison:
batch          1.21
nomal         0.149 - 8.12x slower +5.89 s
```
