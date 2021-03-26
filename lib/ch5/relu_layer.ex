defmodule Ch5.ReluLayer do
  import Nx.Defn
  @default_defn_compiler {EXLA, max_float_type: {:f, 32}}

  defn forward({x, _var}) do
    mask = Nx.greater(x, 0)
    { Nx.multiply(x, mask), mask }
  end

  defn forward_g(x) do
    mask = Nx.greater(x, 0)
    Nx.multiply(x, mask)
  end

  defn backward(x, mask) do
    Nx.multiply(x, mask)
  end
end
