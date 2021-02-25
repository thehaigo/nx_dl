defmodule StepFunc do
  alias Expyplot.Plot
  def draw_graph do
    x = Util.arange(-5.0,5.0,0.1)
    y = Activation.step_function(x)
    Plot.plot([Nx.to_flat_list(x),Nx.to_flat_list(y)])
    Plot.ylim([-0.1,1.1])
    Plot.show()
  end
end
