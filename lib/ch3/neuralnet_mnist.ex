defmodule Ch3.NeruralnetMnist do
  import Nx.Defn
  # comment in exla cpu mode
  @defn_compiler {EXLA, max_float_type: {:f, 64}}
  def get_data() do
    x_test = Dataset.test_image() |> Nx.tensor() |> (& Nx.divide(&1, Nx.reduce_max(&1))).()
    t_test = Dataset.test_label() |> Nx.tensor()
    {x_test, t_test}
  end

  def init_network() do
    {w1,w2,w3,b1,b2,b3} = PklLoad.load("pkl/sample_weight.pkl")
    {
      Nx.tensor(w1),
      Nx.tensor(w2),
      Nx.tensor(w3),
      Nx.tensor(b1),
      Nx.tensor(b2),
      Nx.tensor(b3)
    }
  end

  defn predict(x, {w1,w2,w3,b1,b2,b3}) do
    x
    |> Nx.dot(w1)
    |> Nx.add(b1)
    |> Activation.sigmoid()
    |> Nx.dot(w2)
    |> Nx.add(b2)
    |> Activation.sigmoid()
    |> Nx.dot(w3)
    |> Nx.add(b3)
    |> Activation.softmax()
  end

  def acc do
    {x, t} = get_data()
    IO.puts("Load Weight")
    Benchee.run(%{
      "nomal" => fn -> acc_enum(x, t, init_network()) end,
      "batch" => fn -> acc_enum_batch(x, t, init_network()) end
    })
   end

   def acc_enum(x,t,network) do
     {row, _} = Nx.shape(x)
     Enum.to_list(0..(row - 1))
     |> Nx.tensor()
     |> Nx.map(fn i ->
       predict(x[i],network)
       |> Nx.argmax()
       |> Nx.equal(t[i])
     end)
     |> Nx.sum()
     |> Nx.divide(row)
   end

   def acc_enum_batch(x,t,network) do
     x_b = x |> Nx.to_batched_list(100)
     t_b = t |> Nx.to_batched_list(100)
     {row, _} = x |> Nx.shape()
     batch = Enum.count(x_b)
     Enum.to_list(0..(batch-1))
     |> Nx.tensor()
     |> Nx.map(fn i ->
       predict(Enum.at(x_b,Nx.to_scalar(i)),network)
       |> Nx.argmax(axis: 1)
       |> Nx.equal(Enum.at(t_b,Nx.to_scalar(i)))
       |> Nx.sum()
     end)
     |> Nx.sum()
     |> Nx.divide(row)
   end
end
