defmodule Ch4.SimpleNet do
  def init do
    #Nx.random_normal({2,3})
    Nx.tensor(
      [
        [ 0.47355232, 0.9977393, 0.84668094],
        [ 0.85557411, 0.03563661, 0.69422093]
      ]
    )
  end

  def predict(x,w) do
    Nx.dot(x,w)
  end

  def loss(x,t,w) do
    predict(x,w)
    |> Ch3.Activation.softmax
    |> Ch4.Loss.cross_entropy_error(t)
  end
end
