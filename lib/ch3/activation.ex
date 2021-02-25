defmodule Ch3.Activation do
  import Nx.Defn
  # comment in exla cpu mode
  # @defn_compiler {EXLA, max_float_type: {:f, 64}}
  def step_function(tensor) do
    Nx.map(tensor ,fn t -> if t > 0, do: 1, else: 0 end)
  end

  defn sigmoid(tensor) do
    1 / (1 + Nx.exp(-tensor))
  end

  defn relu(tensor) do
    Nx.max(tensor,0)
  end

  defn softmax(tensor) do
    tensor = Nx.add(tensor, -Nx.reduce_max(tensor))
    tensor
    |> Nx.exp()
    |> Nx.divide( tensor |> Nx.exp() |> Nx.sum())
  end
end
