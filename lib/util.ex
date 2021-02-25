defmodule Util do
  def arange(s, e, f) do
    {step, _} = Float.to_string((e/f) - (s/f)) |> Integer.parse()
    Enum.map(0..step, fn i -> s + (f * i) end) |> Nx.tensor()
  end
end
