defmodule Train do
  alias Ch5.TwoLayerNet, as: Net
  def mini_batch(x_train, t_train, batch_size) do
    row = Enum.count(x_train)
    batch_mask = 0..(row - 1) |> Enum.take_random(batch_size)
    x_batch = Enum.map(batch_mask, fn mask -> Enum.at(x_train, mask) end) |> Nx.tensor |> (& Nx.divide(&1, Nx.reduce_max(&1))).()
    t_batch = Enum.map(batch_mask, fn mask -> Enum.at(t_train, mask) end) |> Nx.tensor
    { x_batch, t_batch }
  end

  def train() do
    x_train = Dataset.train_image
    t_train = Dataset.train_label |> Dataset.to_one_hot
    x_test = Dataset.test_image |> Nx.tensor |> (& Nx.divide(&1, Nx.reduce_max(&1))).()
    t_test = Dataset.test_label |> Dataset.to_one_hot |> Nx.tensor
    IO.puts("data load")
    params = Net.init_params
    table = :ets.new(:grad, [:set, :public])
    iteras_num = 1000
    batch_size = 100
    lr = 0.1
    result = Enum.reduce(
      1..iteras_num,
      %{params: params, loss_list: [], train_acc: [], test_acc: []},
    fn i, acc ->
      IO.puts("#{i} epoch start")
      { x_batch, t_batch } = mini_batch(x_train, t_train, batch_size)
      {grad_w1, grad_b1, grad_w2, grad_b2} = Net.gradient(acc.params, x_batch, t_batch, table)
      {w1, b1, w2, b2} = acc.params

      params = {
        Nx.subtract(w1,Nx.multiply(grad_w1,lr)),
        Nx.subtract(b1,Nx.multiply(grad_b1,lr)),
        Nx.subtract(w2,Nx.multiply(grad_w2,lr)),
        Nx.subtract(b2,Nx.multiply(grad_b2,lr)),
      }

      train_loss_list = [Net.loss_g(params, x_batch, t_batch, batch_size) |> Nx.to_scalar | acc.loss_list ]

      IO.inspect(train_loss_list)
      if (rem(i,batch_size) == 0) do
        IO.inspect(acc.train_acc)
        IO.inspect(acc.test_acc)
        %{
          params: params,
          loss_list: train_loss_list,
          train_acc: [Net.accuracy(params, x_batch, t_batch) |> Nx.to_scalar | acc.train_acc ],
          test_acc: [Net.accuracy(params, x_test, t_test) |> Nx.to_scalar | acc.test_acc]
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
