defmodule GradientCheck do
  def get_dataset do
    x_train = Dataset.test_image |> Nx.tensor |> (& Nx.divide(&1, Nx.reduce_max(&1))).()
    t_train = Dataset.test_label |> Dataset.to_one_hot |> Nx.tensor
    {x_train, t_train}
  end

  def run do
    {x_train, t_train} = get_dataset()
    IO.puts("data load")
    params = Ch5.TwoLayerNet.init_params
    table = :ets.new(:grad, [:set, :public])

    x_batch = x_train[0..99]
    t_batch = t_train[0..99]

    IO.puts("numerical grad")
    grad = Ch5.TwoLayerNet.numerical_gradient(params, x_batch, t_batch)
    IO.inspect(grad)
    IO.puts("gradient")
    prop = Ch5.TwoLayerNet.gradient(params, x_batch, t_batch, table)
    IO.inspect(prop)
    Enum.zip(Tuple.to_list(grad), Tuple.to_list(prop))
    |> Enum.map(fn {g,p} -> Nx.subtract(g,p) |> Nx.abs |> IO.inspect() |> Nx.mean end)
    |> IO.inspect()

    Benchee.run(%{
      "nx grad" => fn -> Ch5.TwoLayerNet.numerical_gradient(params, x_batch, t_batch) end,
      "backpropagation" => fn ->Ch5.TwoLayerNet.gradient(params, x_batch, t_batch, table) end,
    })
  end
end
