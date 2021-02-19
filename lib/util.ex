defmodule Util do
  def arange(s, e, f) do
    {step, _} = Float.to_string((1/f)) |> Integer.parse()
    Enum.map((s * step)..(e * step), fn i -> i / step end) |> Nx.tensor()
  end
end
