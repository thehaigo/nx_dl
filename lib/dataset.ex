defmodule Dataset do
  def download(:mnist) do
    Application.ensure_all_started(:inets)
    file = [
      'train-images-idx3-ubyte.gz',
      'train-labels-idx1-ubyte.gz',
      't10k-images-idx3-ubyte.gz',
      't10k-labels-idx1-ubyte.gz'
    ]
    Mix.shell().cmd("mkdir mnist")
    Enum.each(file, fn f -> get_mnist(f) end)
    :ok
  end

  def get_mnist(file) do
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    {:ok, resp} =
      :httpc.request(:get, {base_url ++ file, []}, [],
        body_format: :binary
      )

    {{_, 200, 'OK'}, _headers, body} = resp

    File.write!("mnist/#{file}", body)
    Mix.shell().cmd("gzip -d mnist/#{file}")
  end

  def train_label() do
    {:ok, <<0, 0, 8, 1, 0, 0, 234, 96, label::binary>>} =
      File.read("mnist/train-labels-idx1-ubyte")
    label |> String.to_charlist()
  end

  def train_image() do
    {:ok, <<0, 0, 8, 3, 0, 0, 234, 96, 0, 0, 0, 28, 0, 0, 0, 28, image::binary>>} =
      File.read("mnist/train-images-idx3-ubyte")
    image |> :binary.bin_to_list() |> Enum.chunk_every(784)
  end

  def test_label() do
    {:ok, <<0, 0, 8, 1, 0, 0, 39, 16, label::binary>>} = File.read("mnist/t10k-labels-idx1-ubyte")

    label |> String.to_charlist()
  end

  def to_one_hot(label) do
    list = label |> Enum.max |> (&(0..&1)).() |> Enum.to_list
    label |> Enum.map(fn t ->  Enum.map(list, fn l -> if t == l, do: 1, else: 0 end) end)
  end

  def test_image() do
    {:ok, <<0, 0, 8, 3, 0, 0, 39, 16, 0, 0, 0, 28, 0, 0, 0, 28, image::binary>>} =
      File.read("mnist/t10k-images-idx3-ubyte")

    image |> :binary.bin_to_list() |> Enum.chunk_every(784)
  end
end
