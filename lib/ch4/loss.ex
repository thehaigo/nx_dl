defmodule Ch4.Loss do
  import Nx.Defn
  # comment in exla cpu mode
  @defn_compiler {EXLA, max_float_type: {:f, 32}}
  defn mean_squared_error(y, t) do
    Nx.power(y-t, 2)
    |> Nx.sum()
    |> Nx.multiply(0.5)
  end

  defn cross_entropy_error(y, t,  batch_size \\ 1.0) do
    Nx.add(y, 1.0e-7)
    |> Nx.log()
    |> Nx.multiply(t)
    |> Nx.sum()
    |> Nx.negate()
    |> Nx.divide(batch_size)
  end
end
