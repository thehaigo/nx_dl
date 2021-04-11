defmodule NxDl.MixProject do
  use Mix.Project

  def project do
    [
      app: :nx_dl,
      version: "0.1.0",
      elixir: "~> 1.10",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  def deps do
    [
      {:axon, "~> 0.1.0-dev", github: "elixir-nx/axon", branch: "main"},
      {:exla, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "exla", override: true},
      {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "nx", override: true},
      {:expyplot, "~> 1.1.2"},
      {:erlport, "~> 0.9.8" },
      {:benchee, "~> 1.0", only: :dev}
    ]
  end
end
