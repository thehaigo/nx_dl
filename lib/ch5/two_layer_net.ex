defmodule Ch5.TwoLayerNet do
  import Nx.Defn
  @default_defn_compiler {EXLA, max_float_type: {:f, 32}}

  defn init_params(input_size \\ 784, hidden_size \\ 100, output_size \\ 10) do
    w1 = Nx.random_normal({input_size, hidden_size}, 0.0, 0.1)
    b1 = Nx.random_uniform({ hidden_size }, 0, 0, type: {:f, 64})
    w2 = Nx.random_normal({ hidden_size, output_size }, 0.0, 0.1)
    b2 = Nx.random_uniform({ output_size }, 0, 0, type: {:f, 64})
    {w1, b1, w2, b2}
  end

  def forward(x, {w1,b1,w2,b2}, table) do
    {x, 0}
    |> Ch5.AffineLayer.forward(w1, b1)
    |> Cache.write(:affine1, table)
    |> Ch5.ReluLayer.forward()
    |> Cache.write(:relu, table)
    |> Ch5.AffineLayer.forward(w2, b2)
    |> Cache.write(:affine2, table)
  end

  def backward(dout, table) do
    dout
    |> Ch5.SoftmaxWithLossLayer.backward(Cache.read(:last, table))
    |> Ch5.AffineLayer.backward(Cache.read(:affine2, table))
    |> Cache.d_write(:affine2d, table)
    |> Ch5.ReluLayer.backward(Cache.read(:relu, table))
    |> Ch5.AffineLayer.backward(Cache.read(:affine1, table))
    |> Cache.d_write(:affine1d, table)
  end

  def predict(x,{_w1,_b1,_w2,_b2} = params,table) do
    forward(x, params, table)
  end

  def loss(x, t, {_w1,_b1,_w2,_b2} = params, batch_size, table) do
    predict(x,params,table)
    |> Ch5.SoftmaxWithLossLayer.forward(t, batch_size)
    |> Cache.write(:last, table)
  end

  defn accuracy({ w1, b1, w2, b2 }, x, t) do
    forward_g(x,{ w1, b1, w2, b2 })
    |> Nx.argmax(axis: 1)
    |> Nx.equal(Nx.argmax(t, axis: 1))
    |> Nx.mean
  end

  def gradient({_w1,_b1,_w2,_b2} = params, x, t, table) do
    # forward
    loss(x, t, params, 100, table)
    # backward
    dout = 1

    backward(dout, table)
    {w1, b1} = Cache.read(:affine1d, table)
    {w2, b2} = Cache.read(:affine2d, table)
    {w1, b1, w2, b2}
  end

  defn forward_g(x, {w1,b1,w2,b2}) do
    x
    |> Ch5.AffineLayer.forward_g(w1, b1)
    |> Ch5.ReluLayer.forward_g()
    |> Ch5.AffineLayer.forward_g(w2, b2)
  end

  defn loss_g({w1,b1,w2,b2}, x, t, batch_size) do
    forward_g(x, {w1,b1,w2,b2})
    |> Ch5.SoftmaxWithLossLayer.forward_g(t, batch_size)
  end

  defn numerical_gradient({ w1, b1, w2, b2 } = params, x, t, batch_size \\ 100) do
    grad(params, &loss_g(&1, x, t, batch_size))
  end
end
