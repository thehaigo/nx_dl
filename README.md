# NxDl

Nx,exlaを使用して以下の書籍を実装したサンプルコードです  
https://www.oreilly.co.jp/books/9784873117584/

https://qiita.com/the_haigo/items/1a2f0b371a3644960251

ch3(with exla)

```
$ iex -S mix
iex(1)> Dataset.download(:mnist)
iex(2)> Ch3.NeruralnetMnist.acc
Name            ips        average  deviation         median         99th %
batch          1.21         0.83 s    ±11.39%         0.83 s         1.00 s
nomal         0.149         6.71 s     ±0.00%         6.71 s         6.71 s

Comparison:
batch          1.21
nomal         0.149 - 8.12x slower +5.89 s
```
