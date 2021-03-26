defmodule Ch5.MulLayer do
  import Nx.Defn

  defn forward(x,y) do
    { Nx.multiply(x, y), {x, y} }
  end

  defn backward({x, y},dout) do
    dx = Nx.multiply(dout, y)
    dy = Nx.multiply(dout, x)
    {dx, dy}
  end
end
