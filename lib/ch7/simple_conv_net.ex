defmodule SimpleConvNet do
  require Axon

  def train_inputs do
    x_train =
      Dataset.train_image
      |> Nx.tensor
      |> Nx.reshape({60000, 1, 28, 28})
      |> (& Nx.divide(&1, Nx.reduce_max(&1))).()
      |> Nx.to_batched_list(100)

    t_train =
      Dataset.train_label
      |> Dataset.to_one_hot
      |> Nx.tensor
      |> Nx.to_batched_list(100)
    {x_train, t_train}
  end

  def test_inputs do
    x_test =
      Dataset.test_image
      |> Nx.tensor
      |> Nx.reshape({10000,1,28,28})
      |> (& Nx.divide(&1, Nx.reduce_max(&1))).()

    t_test = Dataset.test_label |> Dataset.to_one_hot |> Nx.tensor
    {x_test, t_test}
  end

  def model do
    Axon.input({nil,1,28,28})
    |> Axon.conv(30, kernel_size: {5, 5}, activation: :relu)
    |> Axon.max_pool(kernel_size: {2, 2})
    |> Axon.flatten()
    |> Axon.dense(100, activation: :relu)
    |> Axon.dense(10, activation: :softmax)
  end

  def train do
    {x_train, t_train} = train_inputs()
    {trained_params, _optmizer} =
      model()
      |> Axon.Training.step(:categorical_cross_entropy, Axon.Optimizers.adamw(0.005))
      |> Axon.Training.train(x_train, t_train, epochs: 10, compiler: EXLA)

    trained_params
  end

  def test(params) do
    {x_test, t_test} = test_inputs()
    Axon.predict(model(), params, x_test, compiler: EXLA)
    |> Axon.Metrics.accuracy(t_test)
  end
end
