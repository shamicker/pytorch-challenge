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


A formula, ${e}^{i\pi }+1=0$, inside a paragraph.  
A formula, ${e}^{i\pi }+1=0$, inside a paragraph.  
${e}^{i\pi }+1=0$  
${e}^{i\pi }+1=0$  
$y = x^2$
