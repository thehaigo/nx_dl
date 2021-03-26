defmodule Ch5.AffineLayer do
  import Nx.Defn
  @default_defn_compiler {EXLA, max_float_type: {:f, 32}}

  defn forward({x, _var}, w, b) do
    out = Nx.dot(x, w) |> Nx.add(b)
    {out, {w, x}}
  end

  defn forward_g(x, w, b) do
    Nx.dot(x, w) |> Nx.add(b)
  end

  defn backward(dout, {w , x}) do
    dx = Nx.dot(dout, Nx.transpose(w))
    dw = Nx.dot(Nx.transpose(x), dout)
    db = Nx.sum(dout, axes: [0])
    {dx, {dw, db}}
  end
end
