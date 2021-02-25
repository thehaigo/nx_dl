defmodule PklLoad do
  def load(file) do
    { :ok, py_exec } = :python.start([ python_path: 'pkl'])
    result = :python.call( py_exec, :pkl_load, :load, [file])
    :python.stop( py_exec)
    result
  end
end
