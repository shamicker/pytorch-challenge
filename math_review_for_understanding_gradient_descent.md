# Math Review for Understanding Gradient Descent
If you know logarithms, Euler's number `e`, how to calculate derivatives, including the Product Rule, the Quotient Rule, the Power Rule, the Chain Rule, you can probably skip this. 

### This review was mainly done at [Khan Academy](www.khanacademy.org), but also [this article](https://beckernick.github.io/sigmoid-derivative-neural-network/) helped me identify what I needed to learn.

---

#### Derivative as a concept ####

Take the case of a sprinter or runner. 
If the `x-axis` is time and the `y-axis` is the speed, then the slope may be constantly changing. Say we want to calculate the rate of change. This would be the rate of change of `y` as it relates to `x`; the **instantaneous rate of change**, or the rate of change at any given instant.

**This is the definition of __a derivative__ => the slope of a tangent line.

So, for any point `x` on a graphed curve, we want to measure the *steepness*. Basically, we want to measure **the change of `y` as `x` increases**.  

Description of the image below:  
![Delta x over Delta y](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cfrac%7B%5CDelta%20y%7D%7B%5CDelta%20x%7D){.snippet} is the **equation of a slope**, or straight line. Aka **rise over run**.

![delta x over delta y](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cfrac%7B%5Cdelta%20y%7D%7B%5Cdelta%20x%7D){.snippet} is the **equation of a slope of a tangent line**. Super small changes in `y` and super small changes in `x`. Especially as `x` approaches `0`, which just means taking smaller and smaller steps along the curve. (Think of an animated tangent line going from the smallest `x` value to the largest, or from left to right on a graphed curve.) *To find these changes, just use any 2 values of `x` and `y` along that sloped line.*

If `y = f(x)` (said "y equals a function, or f, of x") **describes a curve**,  
then the slope of a tangent line, or the **derivative at any point ![x_1](https://latex.codecogs.com/gif.latex?%5Cinline%20x_%7B1%7D){.snippet} on that curve** can be denoted as ![f prime of x_1](https://latex.codecogs.com/gif.latex?%5Cinline%20%7Bf%7D%27%28x_%7B1%7D%29){.snippet}.

![graphed definition of derivative](images/derivative-definition.png)

#### Three different notations ####
>![3 notations for derivatives](images/derivative-3-notations.png)

##### Lagrange's notation #####
Most common when dealing with functions with a **single variable**.  
`y = f(x)` can also be written as ![y prime](https://latex.codecogs.com/gif.latex?%5Cinline%20%7By%7D%27){.snippet}, which represents the **derivative**.

<!-- ##### Leibniz's notation #####
 -->

![Leibniz's notation explanation](images/leibniz-notation.png){.half1}  

##### Newton's notation #####
- is mostly common in Physics and other sciences where calculus is applied in a real-world context.

##### Derivative as a Slope of a Curve #####
![f prime of 5](https://latex.codecogs.com/gif.latex?%5Cinline%20%7Bf%7D%27%28x%29){.snippet} can be expressed as:  
- slope of the tangent line at `x=5`  
- rate of change of `y` with respect to `x`, of our function `f`

**Remember** that ![Delta x](https://latex.codecogs.com/gif.latex?%5Cinline%20%5CDelta%20x){.snippet} means the *change* or *difference* of `x`. Same thing with the small-d ![small-d delta x](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdelta%20x){.snippet} being a smaller change, or derivative.

![a graphed example](images/change-in-y.png){.quarter}

![another graphed example](images/finding-slope-example.png){.half}

Below is another graphed example. Khan Academy uses `y = mx + b` as the equation for a slope, where `m` is the slope and `b` is the y-intercept. Which also equals: ![point-slope form: y - y_1 = m(x - x_1)](https://latex.codecogs.com/gif.latex?y%20-%20y_%7B1%7D%20%3D%20m%28x%20-%20x_%7B1%7D%29){.snippet}. ![y - y_1](https://latex.codecogs.com/gif.latex?y%20-%20y_%7B1%7D){.snippet} aka ![Delta y](https://latex.codecogs.com/gif.latex?%5CDelta%20y){.snippet}, same with the x's.

In the same graph above, the **derivative** can also be denoted as:  
>![rise over run](https://latex.codecogs.com/gif.latex?%5Cfrac%20%7B%5CDelta%20y%7D%7B%5CDelta%20x%7D){.snippet}  

which is **rise over run**, which equals

>![f prime of x = limit as delta-x approaches 0 is (f(x + delta-x) - f(x))/delta-x](https://latex.codecogs.com/gif.latex?f%5E1%28x%29%20%3D%20%5Clim_%7B%5CDelta%20x%5Crightarrow%200%7D%5Cfrac%20%7Bf%28x%20&plus;%20%5CDelta%20x%29%20-%20f%28x%29%7D%7B%5CDelta%20x%7D){.snippet}

and if ![y = x-squared](https://latex.codecogs.com/gif.latex?y%20%3D%20x%5E%7B2%7D){.snippet}, then:


#### Show derivative of the Sigmoid Function
We'll notice it has a lovely derivative, and it's used more commonly in its derivative notation because it's easier to comprehend than pre-derived!

$\sigma(x) = \frac{1}{1 + e^{-x}}$  

$\sigma'(x) = \frac{\delta}{\delta x}\Big[ \frac{1}{1 + e^{-x}}\Big]$

Using the quotient rule:  $=  \frac{\frac{\delta}{\delta x}\big(1\big)\cdot\big(1 + e^{-x}\big) - \big(1\big)\cdot\frac{\delta}{\delta x}\big(1 + e^{-x}\big)}{\big(1 + e^{-x}\big)^2}$
$= \frac{ 0 - \frac{\delta}{\delta x}(1 + e^{-x})}{(1 + e^{-x})^2}$

Using the basic sum rule:  $= \frac{-\frac{\delta}{\delta x}\big(1\big) + \frac{\delta}{\delta x}\big(e^{-x}\big)}{\big(1 + e^{-x}\big)^2}$
$= \frac{-\frac{\delta}{\delta x}\big(e^{-x}\big)}{\big(1 + e^{-x}\big)^2}$

$= \frac{-\frac{\delta}{\delta x}\big(\frac{1}{e^{x}}\big)}{\big(1 + e^{-x}\big)^2}$ and using the Quotient Rule on the numerator:

$= \frac{\Bigg(\frac{-\frac{\delta}{\delta x}\big(1\big)\cdot\big(e^x\big) - \frac{\delta}{\delta x}\big(e^{x}\big)\cdot\big(1\big)}{\big(e^{2x}\big)}\Bigg)} {\big(1 + e^{-x}\big)^2}$ 

Yes, I know that's complicated, but I think it's easier to understand than if we'd used the Chain Rule rather than the Quotient Rule. (I don't understand which function starts where, for the chain rule, whereas I understand a quotient.)

We know $\frac{\delta}{\delta x}(e^{x}) = e^{x}$. So then

$= \frac{\bigg(\frac{-\big(0\big) - e^{x}}{\big(e^{2x}\big)}\bigg)}{(1 + e^{-x})^2}$

$= \frac{ \bigg(\frac{e^{x}}{e^{2x}}\bigg)} {(1 + e^{-x})^2}$

$= \frac{ \big(\frac{1}{e^{x}}\big)}{(1 + e^{-x})^2}$ and since $\frac{1}{e^{x}}$ is $e^{-x}$, we end up with

$\sigma'(x) = \frac{e^{-x}}{(1 + e^{-x})^2}$  

Phew!!  But this isn't very nice to look at.  
Let's add and subtract 1, which of course doesn't alter anything, but gives us a bit to work with.

$= \frac{+ 1 - 1 + e^{-x}}{(1 + e^{-x})^2}$

You might notice that $(1 + e^{-x})$ is available in both the numerator and the denominator. What if we separate that out?

$= \frac{(1 + e^{-x}) - 1}{(1 + e^{-x})^2}$ equals the same thing as $\frac{(1 + e^{-x})}{(1 + e^{-x})^2} + \frac{-1}{(1 + e^{-x})^2}$

$= \frac{1}{(1 + e^{-x})} - \frac{1}{(1 + e^{-x})^2}$

$\sigma'(x) = \Big(\frac{1}{1 + e^{-x}}\Big)\cdot\Big(1 - \frac{1}{1 + e^{-x}}\Big)$  

And that's it! And since $(\frac{1}{1 + e^{-x}})$ is the original $\sigma(x)$, then we can replace those to become:

$\sigma'(x) = \sigma(x)(1 - \sigma(x))$

Good job! :)

### Calculating the Gradient Descent
The Error formula:

$E = -\frac{1}{m} \sum^{m}_{i=1} \big(y_{i}ln(\hat{y}_{i}) + (1 - y_{i})ln(1 - \hat{y}_{i}) \big)$

To calculate the gradient of E (the Error), at a specific point x, it's $\Delta E = \big(\frac{\delta}{\delta w_{1}}E, \frac{\delta}{\delta w_{2}}E, ..., \frac{\delta}{\delta w_{n}}E, \frac{\delta}{\delta b}E\big)$.

It's the error that each point produces, and calculate the derivative of this error. The TOTAL error is just the average of error at all points. ie, $E = -yln(\hat{y}) - (1-y)ln(1-\hat{y})$.

Anyways, skipping some explanations, we get that the **derivative of the Error** at a point `x` with respect to weight `w_{j}` is $-(y - \hat{y})x_{j})$, and with respect to `b` is $-(y - \hat{y})$.

**Which all shows** that, at a coordinate x, the gradient of the Error function is $\Delta E = -(y - \hat{y})(x_{1}, x_{2}, ..., x_{n}, 1)$.

And **THIS all means** that the gradient is actually a **scalar** times the coordinates of the point. And the **scalar** is a multiple of the difference between `y` and `\hat{y}`.

Which means the smaller the prediction, the smaller the gradient. And the bigger the prediction, the bigger the gradient. 

Which means the higher up in the mountains we are, the bigger the steps we need to take to get down. And presumably, smaller steps nearer the bottom.

