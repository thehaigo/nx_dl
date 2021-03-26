defmodule Ch5.SoftmaxWithLossLayer do
  import Nx.Defn
  @default_defn_compiler {EXLA, max_float_type: {:f, 32}}

  defn forward({x, _var}, t, batch_size \\ 1) do
    y = Ch3.Activation.softmax(x)
    loss = Ch4.Loss.cross_entropy_error(y,t, batch_size)
    {loss, {y, t}}
  end

  defn forward_g(x, t, batch_size) do
    Ch3.Activation.softmax(x)
    |> Ch4.Loss.cross_entropy_error(t, batch_size)
  end

  defn backward(_, {y, t}) do
    {batch_size, _} = Nx.shape(t)
    Nx.subtract(y,t)
    |> Nx.divide(batch_size)
  end
end
