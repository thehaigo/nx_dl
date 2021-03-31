defmodule ThreeLayerNetwork do
  def init_network do
    %{}
    |> Map.put(:w1, Nx.tensor([[0.1,0.3,0.5],[0.2,0.4,0.6]]))
    |> Map.put(:b1, Nx.tensor([[0.1,0.2,0.3]]))
    |> Map.put(:w2, Nx.tensor([[0.1,0.4],[0.2,0.5],[0.3,0.6]]))
    |> Map.put(:b2, Nx.tensor([[0.1,0.2]]))
    |> Map.put(:w3, Nx.tensor([[0.1,0.3],[0.2,0.4]]))
    |> Map.put(:b3, Nx.tensor([[0.1,0.2]]))
  end

  def forward(x,network) do
    [w1, w2, w3] = [network.w1, network.w2, network.w3]
    [b1, b2, b3] = [network.b1, network.b2, network.b3]

    x
    |> Nx.dot(w1)
    |> Nx.add(b1)
    |> Ch3.Activation.sigmoid()
    |> Nx.dot(w2)
    |> Nx.add(b2)
    |> Ch3.Activation.sigmoid()
    |> Nx.dot(w3)
    |> Nx.add(b3)
  end
end
