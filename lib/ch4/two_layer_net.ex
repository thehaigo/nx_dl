defmodule Ch4.TwoLayerNet do
  import Nx.Defn
  @default_defn_compiler {EXLA, max_float_type: {:f, 32}}

  defn init_params(input_size \\ 784, hidden_size \\ 100, output_size \\ 10) do
    w1 = Nx.random_normal({input_size, hidden_size}, 0.0, 0.1)
    b1 = Nx.random_uniform({ hidden_size }, 0, 0, type: {:f, 64})
    w2 = Nx.random_normal({ hidden_size, output_size }, 0.0, 0.1)
    b2 = Nx.random_uniform({ output_size }, 0, 0, type: {:f, 64})
    { w1, b1, w2, b2 }
  end

  defn predict({ w1, b1, w2, b2 }, x) do
    x
    |> Nx.dot(w1)
    |> Nx.add(b1)
    |> Ch3.Activation.sigmoid
    |> Nx.dot(w2)
    |> Nx.add(b2)
    |> Ch3.Activation.softmax
  end

  defn loss({ w1, b1, w2, b2 }, x, t, batch_size) do
    predict({w1, b1, w2, b2}, x)
    |> Ch4.Loss.cross_entropy_error(t, batch_size)
  end

  defn accuracy({ w1, b1, w2, b2 }, x, t) do
    predict({ w1, b1, w2, b2 }, x)
    |> Nx.argmax(axis: 1)
    |> Nx.equal(Nx.argmax(t, axis: 1))
    |> Nx.mean
  end

  defn numerical_gradient({ _w1, _b1, _w2, _b2 } = params, x, t, batch_size) do
    grad(params, loss(params, x, t, batch_size))
  end

  defn update({ w1, b1, w2, b2 } = params, x, t, lr, batch_size) do
    {grad_w1, grad_b1, grad_w2, grad_b2} = grad(params, &loss(&1, x, t, batch_size))

    {
      w1 - (grad_w1 * lr),
      b1 - (grad_b1 * lr),
      w2 - (grad_w2 * lr),
      b2 - (grad_b2 * lr)
    }
  end

  def mini_batch(x_train, t_train, batch_size) do
    row = Enum.count(x_train)
    batch_mask = 0..(row - 1) |> Enum.take_random(batch_size)
    x_batch = Enum.map(batch_mask, fn mask -> Enum.at(x_train, mask) end) |> Nx.tensor |> (& Nx.divide(&1, Nx.reduce_max(&1))).()
    t_batch = Enum.map(batch_mask, fn mask -> Enum.at(t_train, mask) end) |> Nx.tensor
    { x_batch, t_batch }
  end

  def train(params) do
    x_train = Dataset.train_image
    t_train = Dataset.train_label |> Dataset.to_one_hot
    x_test = Dataset.test_image |> Nx.tensor |> (& Nx.divide(&1, Nx.reduce_max(&1))).()
    t_test = Dataset.test_label |> Dataset.to_one_hot |> Nx.tensor
    IO.puts("data load")

    iteras_num = 1000
    batch_size = 100
    lr = 0.1
    result = Enum.reduce(
      1..iteras_num,
      %{params: params, loss_list: [], train_acc: [], test_acc: []},
    fn i, acc ->
      IO.puts("#{i} epoch start")
      { x_batch, t_batch } = mini_batch(x_train, t_train, batch_size)
      params = update(acc.params, x_batch, t_batch, lr, batch_size)
      train_loss_list = [loss(acc.params, x_batch, t_batch, batch_size) |> Nx.to_scalar | acc.loss_list ]

      IO.inspect(train_loss_list |> Enum.reverse)
      if (rem(batch_size, i) == 0) do
        IO.inspect(acc.train_acc)
        IO.inspect(acc.test_acc)
        %{
          params: params,
          loss_list: train_loss_list,
          train_acc: [accuracy(params, x_batch, t_batch) |> Nx.to_scalar | acc.train_acc ],
          test_acc: [accuracy(params, x_test, t_test) |> Nx.to_scalar | acc.test_acc]
        }
      else
        %{
          params: params,
          loss_list: train_loss_list,
          train_acc: acc.train_acc,
          test_acc: acc.test_acc
        }
      end
    end)
    IO.inspect(result.loss_list)
  end
end
