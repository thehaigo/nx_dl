defmodule Ch4.Grad do
  import Nx.Defn
  # comment in exla cpu mode
  # @defn_compiler {EXLA, max_float_type: {:f, 64}}
  def numerical_diff(func \\ &nop/1, tensor) do
    h = 1.0e-4
    func.(Nx.add(tensor, h))
    |> Nx.subtract(func.(Nx.subtract(tensor, h)))
    |> Nx.divide(2 * h)
  end

  def numerical_gradient(func \\ &nop/1 , tensor) do
    h = 1.0e-4
    Enum.to_list(0..(Nx.size(tensor) - 1))
    |> Enum.map(fn i ->
      idx_p =
        tensor
        |> Nx.to_flat_list
        |> List.update_at(i, &(&1 + h))
        |> Nx.tensor()
        |> Nx.reshape(Nx.shape(tensor))

      idx_n = tensor
        |> Nx.to_flat_list
        |> List.update_at(i, &(&1 - h))
        |> Nx.tensor()
        |> Nx.reshape(Nx.shape(tensor))

      func.(idx_p)
      |> Nx.subtract(func.(idx_n))
      |> Nx.divide(2 * h)
      |> Nx.to_flat_list
    end)
    |> Nx.tensor()
    |> Nx.reshape(Nx.shape(tensor))
  end

  def gradient_descent(f, tensor, lr \\ 0.01, step_num \\ 100) do
    Enum.reduce(
      0..(step_num - 1),
      tensor,
      fn _, acc ->
        numerical_gradient(f, acc)
        |> Nx.multiply(lr)
        |> Nx.subtract(acc)
      end
    )
  end

  defp nop(enum), do: enum
end
