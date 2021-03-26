defmodule Ch5.AddLayer do
  import Nx.Defn

  defn forward(x,y) do
    {Nx.add(x, y), {x, y}}
  end

  defn backward({_x, _y}, dout) do
    {dout * 1, dout * 1}
  end
end
