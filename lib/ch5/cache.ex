defmodule Cache do
  def write({out, var}, key, table) do
    :ets.insert(table, { key, var })
    {out, 0}
  end

  def read(key, table) do
    :ets.lookup(table, key)[key]
  end

  def d_write({dout, var}, key, table) do
    :ets.insert(table, { key, var })
    dout
  end
end
