defmodule BuyAppleOrange do
  alias Ch5.{ MulLayer, AddLayer }
  def main do
    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3
    tax = 1.1

    {apple_price, mul_apple_layer} = MulLayer.forward(apple, apple_num)
    {orange_price, mul_orange_layer}= MulLayer.forward(orange, orange_num)
    {all_price, add_apple_orange_layer} = AddLayer.forward(apple_price, orange_price)
    {price, mul_tax_layer} = MulLayer.forward(all_price, tax)

    dprice = 1
    {dall_price, dtax} = MulLayer.backward(mul_tax_layer, dprice)
    {dapple_price, dorange_price} = AddLayer.backward(add_apple_orange_layer, dall_price)
    {dorange, dorange_num} = MulLayer.backward(mul_orange_layer, dorange_price)
    {dapple, dapple_num} = MulLayer.backward(mul_apple_layer, dapple_price)
    IO.inspect(price)
    IO.inspect({dapple_num, dapple, dorange, dorange_num, dtax})

  end
end
