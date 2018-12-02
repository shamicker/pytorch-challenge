# PyTorch Scholarship Challenge
Start date: Nov 9, 2018  
End date: Jan 9, 2019

## Welcome to the Scolarship Challenge!
Welcome to the course! Here we'll get you oriented and set you up for success in this course.  
ETA 30 minutes

### Course Overview
Deep learning:  
- within the field of machine learning, using massive neural networks, massive datasets, and accelerated computing on GPU  
- impacts many industries already:  
   - personal voice assistants  
   - medical imaging  
   - automated vehicles  
   - video game AI  
   - etc  

### PyTorch
- is open-source Python framework from Facebook's AI research team
- used for developing deep neural networks

### Course Outline
1. Introduction to Neural Networks  
  - Learn the concepts behind deep learning and how we train deep neural networks with backpropagation.
2. Talking PyTorch with Soumith Chintala  
  - Cezanne Camacho and Soumith Chintala, the creator of PyTorch, chat about the past, present, and future of PyTorch.
3. Introduction to PyTorch  
  - Learn how to build deep neural networks with PyTorch  
  - Build a state-of-the-art model using a pre-trained network that classifies cat and dog images
4. Convolutional Neural Networks  
  - Here you'll learn about convolutional neural networks, powerful architectures for solving computer vision problems  
  - Build and train an image classifier from scratch to classify dog breeds
5. Style Transfer  
  - Use a trained network to transfer the style of one image to another image  
  - Implement the style transfer model from Gatys et al.
6. Recurrent Neural Networks  
  - Learn how to use recurrent neural networks to learn from sequences of data such as time series  
  - Build a recurrent network that learns from text and generates new text one character at a time
7. Sentiment Prediction with an RNN  
  - Build and train a recurrent network that can classify the sentiment of movie reviews
8. Deploying PyTorch Models  
  - Learn how to use PyTorch's Hybrid Frontend to convert models from Python to C++ for use in production

And we'll be ending with building this from scratch!!!  

![challenge project](pytorch-project-image.png){.center}
<style>
  .markdown-body .center {
    display: block;
    border: 1px solid silver;
    text-align: center;
  }
</style>

## Introduction to Neural Networks
Learn the concepts behind how neural networks operate and how we train them using data.  
ETA 2 hours

### Introduction: What is Deep Learning and what is it used for?
Used for:  
- beating humans in games like Go and Jeopardy  
- detecting spam in emails  
- forecasting stock prices  
- recognizing images in a picture  
- diagnosing illnesses  
- self-driving cars

Neural Network:  
- imitates how a brain works  
- basically, finds a boundary between different categories (like different flower species)

### Classification Problems 1
**A simple example:**
We know that:  
- student A: 9/10 on a test, 8/10 grades - passes the class  
- student B: 3/10 on test, 4/10 grades - fails the class  
Does student C pass or fail? S/he has 7/10 on test, 6/10 grades? One way to find out is to plot the students' marks:
![plot marks on a graph](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/classification-ex.PNG){.half1}
~~(above) Plot of the first 2 students~~

![plot all known marks](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/classification-ex2.PNG){.half}
~~Plot the whole class~~

![divide categories (classify them)](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/classification-ex2-answer.PNG){.half}
~~The best linear boundary to divide the Passes from the Fails~~
<style>
  .markdown-body{width: 90%;}
  .half1{width:62%;}
  .half{width:60%;}
  img + del { 
    text-decoration: none;
    font-style: italic;
    display: block;
    text-indent: 7%;
  }
</style>

Great! Well, this boundary is easy for us to visualize, but not easy for a computer.

### Linear Boundaries
This boundary is linear; ie. a single, straight line. Calculated by ![w_{1}x_{1} + w_{2}x_{2} + b = 0](https://latex.codecogs.com/gif.latex?w_%7B1%7Dx_%7B1%7D%20&plus;%20w_%7B2%7Dx_%7B2%7D%20&plus;%20b%20%3D%200){.snippet}.
<style>
  .snippet {
    /*background-color: #eee;*/
    padding: 0;
    padding-top: 0.2em;
    padding-bottom: 0.2em;
    margin: 0;
    font-size: 85%;
    background-color: rgba(0,0,0,0.04);
    border-radius: 3px;
    font-family: Consolas, "Liberation Mono", Menlo, Courier, monospace;
  }
</style>

For a computer, we need an equation to create this boundary.
![linear boundary equation for this example](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/equation.PNG)

A general linear boundary equation is ![w_{1}x_{1} + w_{2}x_{2} + b = 0](https://latex.codecogs.com/gif.latex?w_%7B1%7Dx_%7B1%7D%20&plus;%20w_%7B2%7Dx_%7B2%7D%20&plus;%20b%20%3D%200){.snippet}, abbreviated in vector notation as ![abbrev equation](https://latex.codecogs.com/gif.latex?Wx%20&plus;%20b%20%3D%200){.snippet}, where `W`{.snippet} is the vector ![vector W](https://latex.codecogs.com/gif.latex?W%20%3D%20%28w_%7B1%7D%2C%20w_%7B2%7D...%29){.snippet}, *x* is the vector ![vector x](https://latex.codecogs.com/gif.latex?x%20%3D%20%28x_%7B1%7D%2C%20x_%7B2%7D...%29){.snippet}, and the equation is *W* times *x* plus *b*. We'll refer to *x* as the **inputs**, *W* as the **weights**, and *b* as the **bias**.

*y* is the label that we're trying to predict. It'll be either 1 or 0 (ie, true or false, in this case); 1 if *y* is above the line (ie ![greater than or equal to 0](https://latex.codecogs.com/gif.latex?%5Cgeq%200){.snippet}), or 0 if *y* is below the line (ie < 0).

Our prediction will be *"y hat equals 1 if above the line, or y hat equals 0 if below the line"*:  
![y hat = 1 or 0](https://latex.codecogs.com/png.latex?%5Cinline%20%5Chat%7By%7D%20%3D%20%5Cbegin%7BBmatrix%7D%20%5Cbegin%7Balign*%7D%20%26%201%20if%20Wx&plus;b%5Cgeq0%20%5C%5C%20%26%200%20if%20Wx&plus;b%3C0%20%5Cend%7Balign*%7D%5Cend%7BBmatrix%7D){.snippet}.

***NOTE:*** y is the ACTUAL solution vector, while ![\hat{y}](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Chat%7By%7D){.snippet} is the PREDICTED solution.

![y hat = 1 or 0](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/y-hat.PNG)

### Higher Dimensions
Higher dimensional boundaries are needed when there are more data points to look at.  
For instance, a plane boundary (a flat surface) occurs when there is ![w1x1 + w2x2 + w3x3 + b = 0](https://latex.codecogs.com/gif.latex?%5Cinline%20w_%7B1%7Dx_%7B1%7D%20&plus;%20w_%7B2%7Dx_%7B2%7D%20&plus;%20w_%7B3%7Dx_%7B3%7D%20&plus;%20b%20%3D%200){.snippet}.  
There are also even higher dimensions with ![w1x1 + w2x2 + ... + wnxn + b = 0](https://latex.codecogs.com/gif.latex?%5Cinline%20w_%7B1%7Dx_%7B1%7D%20&plus;%20w_%7B2%7Dx_%7B2%7D%20&plus;%20...%20&plus;%20w_%7Bn%7Dx_%7Bn%7D%20&plus;%20b%20%3D%200){.snippet}.

### Perceptrons
- the building block of neural networks
- just an encoding of our equation into a small graph:  
   - each input (x) is in a node  
   - the weights (w) are labeled on the *edges* (the arrows) of the input nodes  
   - the bias (b) can either be in the calculation node or an "input" node  
   - the calculation is also in a node  
   - the prediction (score or y-hat) is the return value

![perceptron using our example](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/perceptron-example.PNG){.half}
![perceptron with summing equation](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/perceptron-summation.PNG){.half}

In our example, we're using an implicit function called a **step function**. This is sometimes viewed in its own node, in which case the *linear function* is in a node where the calculation occurs ("the first node calculates the linear equation on the inputs of the weights"), and the *step function* is in a node where the step function is applied to the result of the calculation.

![perceptron showing both the linear function and the step function](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/perceptron-linear-stepfunction.PNG){.half}

In the future, we'll be using different step functions, so it is useful to specify it in the node.

### Why 'Neural Networks'? ###
Why are they called neural networks?

In the same way the *input* nodes are one-way to the calculation node, resulting in a one-way output, a neuron's *dendrites* give a one-way input to the *nucleus*, where something is decided, and a one-way electrical impulse is sent (or not?) out through the *axon*.

So far, we're working with a single "neuron". Later it'll get more complicated with whole networks of these calculations!

![perceptron vs neuron](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/neurons.PNG){.half}

### Perceptrons as Logical Operators ###
Perceptrons can be used as logical operators as well.  
![an AND perceptron](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/perceptron-and.PNG){.quarter}
<style type="text/css">
  .quarter { width: 30%; }
</style>

#### AND Perceptron
QUIZ: What are the weights and bias for the AND perceptron?  
More than 1 set of values will work!  
![an AND perceptron visualized](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/and-perceptron.PNG){.half}

#### OR Perceptron
QUIZ: What are the weights and bias for the OR perceptron?  
Basically, the weights work together to create the angle of the linear boundary (think of it like a teeter-totter where different weights on the x's change the angle) and the bias (actually it's the *difference* between the bias and the weighted inputs) sort of moves that boundary left or right. SO to move the boundary over (while maintaining the same angle), you can either change the weights (by the same amount) OR change the bias.  
![an OR perceptron visualized](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/perceptron-or.PNG){.half}

#### NOT Perceptron ####
The **NOT** only cares about a single input (like, for input (0, 1) it cares only about either the 0 or the 1); all other inputs are ignored. This single input is then turned into the opposite.

#### XOR Perceptron ####
The **XOR** perceptron returns a positive value ONLY when the inputs are different (I think?).  
![an XOR perceptron visualized](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/perceptron-xor.PNG){.half}

As you can see, it's not the regular **OR** since the first input (1,1) returns a false value! To get around this, we have to create a *multi-layer perceptron* or a (very basic) neural network. 

**NAND** is short for **NOT AND**.  

![an XOR neural network](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/xor-neuralnetwork.PNG){.smaller_half} 
![an XOR multi-layer perceptron](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/xor-multi-layer-perceptron.PNG){.smaller_half}
<style type="text/css">
  .smaller_half {width: 45%;}
</style>

### Perceptron Trick ###
Since we can't calculate the linear boundary by hand every time, we can get a computer to find the equation. How do we do that? Using the Perceptron Trick!  

The computer begins with a random equation (random line). The correctly classified data points say "I am good!", and the incorrect points say "Come closer!". Obviously the computer doesn't understand English that way (and nor do the points), so it's using an adjustment equation.

![perceptron trick cartoon](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/perceptron-trick.PNG)

Example: A random equation of ![3x1 + 4x2 - 10 = 0](https://latex.codecogs.com/gif.latex?%5Cinline%203x_%7B1%7D%20&plus;%204x_%7B2%7D%20-%2010%20%3D%200){.snippet}. If we have a point (1,1) inside the current negative area that should be in the positive area, the computer should add the data point to the equation (with 1 as the bias). However, since this drastically changes the equation and might be overkill, we use a ***learning rate***. In this case it'll be 0.1, so we only add a 10th of the incorrectly classified data point. So the new equation becomes ![3.1x_1 + 4.1x_2 - 9.9 = 0](https://latex.codecogs.com/gif.latex?%5Cinline%203.1x_%7B1%7D%20&plus;%204.1x_%7B2%7D%20-%209.9%20%3D%200){.snippet}.

The computer continues doing this until it correctly classifies all (or, more probably, most) of the data points.  In this example, it would do this 10 times to get the point (1,1) into the positive area.

**NOTE:** If a data point is incorrectly classified in the positive area (meaning the line is "more positive" than it should be, since the points don't move but the line does), a 10th of that data point should be *subtracted* from the equation in order to make the line move towards the negative.

### Perceptron Algorithm ###
The algorithm is just the equation in "math" terms. The ![alpha](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Calpha){.snippet} is the *learning rate*, and coordinates are `(p, q)`.

Holy cow, it took me a long time to get this! Mostly because I didn't understand that the coordinate (p, q) is an element in x - as in ![x1](https://latex.codecogs.com/gif.latex?%5Cinline%20x_%7B1%7D){.snippet}, ![x2](https://latex.codecogs.com/gif.latex?%5Cinline%20x_%7B2%7D){.snippet}, and ![x3](https://latex.codecogs.com/gif.latex?%5Cinline%20x_%7B3%7D){.snippet} are each a coordinate `(p, q)`! I won't write it here since it was only a small portion of the whole code. I'm sure I'll be able to get it by the end of this Challenge!

### Non-Linear Regions ###
There are also data sets that are more complex and cannot be divided linearly.  
So next, we're looking at a curve!

### Error Functions ###
We'll be using **error functions** to tell us how far we are from the solution - just the distance from the goal.

### Log-loss Error Function ###
I didn't fully understand this. **Log-loss** wasn't mentioned and the quiz was for *gradient descent* which he said we'd look at later...?

Someone in the forums (Rusty) said:

```
log loss = 2 possible outputs
cross entropy = 3 or more possible outputs

log loss = binary cross entropy
```

Basically, rather than discrete distances (like 2, 3, 4), it's better to have continuous distances (like 9.00, 8.57, 3.90) so we can orient ourselves better.

Also, penalties (distance points) are included but minimal for *correctly* classified data points, as well as being larger for incorrectly classified data.

### Discrete vs Continuous ###
Currently, our algorithm (linear boundary) is discrete (the answer is either Yes (1) or No (0)). We need to convert it to a continuous outcome. So we'll change from a **step function** to a **sigmoid function**.

>![activation functions](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/activation_functions.PNG){.half}

A *sigmoid function* gives us a probability of passing. The high end approaches 1, while the negative approaches 0. The middle is 50%, which is when the linear boundary `Wx + b` equalled 0.

![activation function predictions](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/activation_function_predictions.PNG){.half1}  
![sigmoid function formula](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/sigmoid_function.PNG){.quarter}

```
@Zenquiorra (on slack)
W and X are an array of values, their multiplication(linear combination) will give a scalar value (say 'x'). This x, when substituted in 1/(1+e^(-x)), will find the value of the sigmoid function of Wx+b
```

### Softmax ###
So far, we've only been talking binary predictions. What about if we want more classifications? `[cat, dog, bird]` for example?

The answer is the **Softmax function**.   
Example: The outcomes are either:  
a duck - at a score of 2  
a beaver - at a score of 1  
a walrus - at a score of 0

We tried: `2/(2+1+0)` for the duck, `1/(2+1+0)` for the beaver, etc. However, if we have negative outcomes, it's pretty easy to end up trying to divide by zero!

A function that turns every (including negative) number into a positive is the exponential. We'll use ![e to the x](https://latex.codecogs.com/gif.latex?%5Cinline%20e%5Ex){.snippet}. The "same" formula then becomes ![e^2/e^2 + e^1 + e^0)](https://latex.codecogs.com/gif.latex?%5Cinline%20e%5E2/%28e%5E2%20&plus;%20e%5E1%20&plus;%20e%5E0%29){.snippet} for the duck, ![e^1/e^2 + e^1 + e^0)](https://latex.codecogs.com/gif.latex?%5Cinline%20e%5E1/%28e%5E2%20&plus;%20e%5E1%20&plus;%20e%5E0%29){.snippet} for the beaver, and ![e^0/e^2 + e^1 + e^0)](https://latex.codecogs.com/gif.latex?%5Cinline%20e%5E0/%28e%5E2%20&plus;%20e%5E1%20&plus;%20e%5E0%29){.snippet} for the walrus. These functions give us a beautiful probability for our 3 outcomes: `0.67 duck`, `0.24 beaver`, `0.09 walrus`.

The **Softmax Function** is defined as:

For linear function scores: ![Z1, Z2, ... Zn](https://latex.codecogs.com/gif.latex?%5Cinline%20Z_%7B1%7D...Z_%7Bn%7D){.snippet} (each is a score for each of the classes), the probability (P) that the object in some class i is (as follows):

>![probability of class i](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/softmax_function.PNG){.quarter}

### One-Hot Encoding ###
If you only have 2 input variables, it's easy to numerize them: it's `1` or `0`.  
But if you have multiple inputs, you can't put `0, 1, 2`, because that assumes a dependence between classes (ie a hierarchy or something). What's the solution?

The solution is kind of like binary; it's an identity matrix, which is the matrix equivalent of 1. (Like, you can multiply ANY matrix by an identity matrix and you'll get the original matrix back.)  
> ![it's an identity matrix](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/math_images/identity_matrix.PNG){.half}

![one-hot encoding](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/one-hot_encoding.PNG){.smaller_half}

### Maximum Likelihood ###
Given any 2 linear boundaries in an example model (say, our students' model), which line is a better line? Here is where we'll need an equation to calculate each line's probability.

**Method:** multiply all the y_hat values together to get the total probability (of all combined data points).  
![total probability examples](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/maximum_likelihood.PNG){.half}

### Maximizing Probabilities ###
Remember that we're talking about **Error Function**, and that we want to *minimize* the error function. And also we want to *maximize* the probability. So maybe these 2 work together, or are even the same thing.

However! Products are tough - and for potentially thousands of numbers between 0 and 1, there will be TONS of decimal points!  
Instead, we want to take sums and *turn them into products*!!  

It turn out that:
>![log of (ab) = log of a + log of b](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/log_for_sums.PNG){.half}

#### Logarithm Reminder ####
>![log explanation](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/log_defined.PNG){.half}


![a log from a tree](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/log.jpg){.one_third} == ![an image of Rosie as a witch](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/witch_power.jpg){.one_third}  
<style type="text/css">
  .one_third{
    width: 15%;
  }
</style>
Maybe easier to remember: "log" == "to which power?"  

### Cross-Entropy 1 ###
"From now until the end of class, we'll be taking the natural logarithm,  which is base e instead of 10. Nothing different happens to base 10; everything works the same cuz everything gets scale by the same factor so it's just more for convention."  

Where **ln** is logarithm:
>![the product in our examples vs the natural logarithm](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/base_e.PNG){.eighty}
<style type="text/css">
  .eighty { width: 80%; }
</style>

Notice all of them are negative numbers. The log of anything between 0 and 1 is a negative number - the log of 1 = 0 (and log of 0 is undefined since no number squared can equal 0). 

We know that a high probability of success approaches 1 (which is 100%). When we take the *negative logarithm* of that probability, we get a low number (it's how logs work). We can therefore think of this resulting *negative log* as errors at each point since low will be good scores.

*To get positive numbers, we'll take the **negative** of the log of the probabilities.* This is called **Cross-Entropy**.

So now our goal has changed from maximizing the probability to minimizing the cross entropy (errors). 


### Cross-Entropy 2 ###
If we put our weights as `1` for the higher (of 2) probability of pass/fail output per input, and `0` for the less likely probability. For example, if the probability of finding a gift is 0.8 (yes or `1`) and 0.2 for not finding one (no or `0`). **Note that the probabilities for each data point need to add up to 1 (0.8 + 0.2 == 1.0)**

For each data point (input), we add the product of the weight and `-log`, and the take the sum of each of those sums.

`input 1` = `(yes)(-log(input 1a)` + `(no)(-log(input 1b)`  
`input 2` = `(yes)(-log(input 2a)` + `(no)(-log(input 2b)`  
and then `input 1 + input 2` equals the **cross entropy**.

**Note** that `no` can be written as `0`, so `1-0` is also valid.  
**Note** that `input b` is `1 - input a` since inputs a and b add up to 1!

>![cross entropy example and formula](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/cross-entropy.PNG){.half1}

### Multi-Class Cross Entropy ###
This formula works for multiple unrelated classes.
![multiple classed cross entropy example](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/multiclass-crossentropy.PNG){.half1}

Where:
`p` is the probability  
`j` is the door  
`n` is the number of doors, starting at `i`  
`m` is the number of animals (or *classes*), starting at `j`

![multi-class cross entropy formula and example](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/multiclass-crossentropy2.PNG){.half1}

### Logistic Regression ###
**Logistic Regression**: "the building block of all that constitutes Deep Learning!"  

Basically:
- take your data
- pick a random model
- calculate the error
- minimize the error and obtain a better model
- enjoy!

To recap the cross entropy function:  
![error function explanation](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/error-function-pre-formula.PNG)

The difference between the *cross-entropy* and the *error* functions is just that we **take the average of the cross-entropy function** to get the error function.

And in terms of weights `W` and bias `b`:  
![error function in terms of W and b](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/error-function.PNG)

Difference between binary and multiclass formulas: (images give class types)
![2-class and multi-class formulas](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/2-to-many-class-entropy-formulas.PNG)

### Gradient Descent ###
If you need a math review for this, you can find it [at this spot](math_review_for_understanding_gradient_descent.md) or [here on my computer...](file:///C:/users/shauna/appdata/local/temp/16.html).

The derivative of the Sigmoid Function is really nice, and it is used more often in some cases (ie backpropagation). More on this later.

The Sigmoid Function:  $\sigma(x) = \frac{1}{(1 + e^{-x})}$

Its derivative:  $\sigma'(x) = \sigma(x)(1 - \sigma(x))$

Now, the **Gradient Descent** in mathematical terms:

The Error Function is a function of the *weights*. 

![error function as "mount math-er-horn"](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/error_function_gradient_descent.PNG)

The gradient of E is given by the **partial derivatives** with respect to $w_{1}$ and $w_{2}$. Wherever we're "standing", we'll take the **negative** of the gradient of the Error function at that point, repeatedly until we're all good.

Where the gradient is **the direction of steepest ascent** (see image below, where the blue arrows are shortest and the red ones are longest), we will take the **negative** of that.

![direction of steepest ascent in an image](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/math_images/gradient_definition.PNG)

In this image/example, we want to go down the mountain in the quickest way possible. We look around us, calculate a small/safest way down, then look around and recalculate, step down again, etc.

So we have an initial prediction $\hat{y} = \sigma(Wx + b)$, which is "bad" because we are currently still really high in the mountains. We take the *gradient* of the Error function, which is the **vector produced** by the partial derivatives of the Error function, with respect to the weights and the bias. 

Now we take a step in the direction of the *negative* of the gradient. Since we don't want any dramatic changes, we have a **learning rate**. And we update the weights and bias: we take the *learning rate* times the partial derivative of the Error, with respect to the weights/bias and add them respectively.

In other words, we are taking the steepest step down (found by the *negative of the gradient*), and then updating our position to a fraction less steep (found by subtracting a fraction (ie the *learning rate*) of the steepness vector). And we repeat until we have the best outcome, which is us at the lowest point in the mountains.

To reiterate slightly mathematically, at a coordinate x, the gradient of the Error function is $\Delta E = -(y - \hat{y})(x_{1}, x_{2}, ..., x_{n}, 1)$.

This means that the gradient is actually a **scalar** times the coordinates of the point. And the **scalar** is a multiple of the difference between $y$ and $\hat{y}$.

Which means the smaller the prediction, the smaller the gradient. And the bigger the prediction, the bigger the gradient. 

Which means the higher up in the mountains we are, the bigger the steps we need to take to get down. And presumably, smaller steps nearer the bottom.

### Logistic Regression Algorithm
So the pseudo-code:  
- With random starting weights: (w_{1}, ... w_{n}, b)  
- For every point in our coordinate x we calculate the error:  
    - For each point:  
        - Update the weights: $w'_{i} \gets w_{i} - \alpha(\hat{y} - y)x_{i}$  
        - Update the bias: $b'_{i} \gets b_{i} - \alpha(\hat{y} - y)$  
- Repeat either a fixed number of times, or until the error is small.

This looks suspiciously similar to the Perceptron Algorithm.

### Pre-Notebook: Gradient Descent
This was just instructions to do the next step.
### Notebook: Gradient Descent
This was a "homework" task: Basically, like the Perceptron Algorithm, code the Logistic Regression Algorithm steps in Python.

### Perceptron vs Gradient Descent
If you compare these two calculations, you'll notice they're the same. 

#### Perceptron Algorithm
We updated the weights and the bias ONLY if they were mis-classified. Also, $y$ and $\hat{y}$ were ONLY `1` or `0`.

#### Gradient Descent Algorithm (logistic regression)
We updated ALL the weights and the bias whether they were misclassified or not. Also, $y$ and $\hat{y}$ are a range between `0` and `1`. If the points are correctly classified, it's telling the line to go farther away.

### Continuous Perceptrons
Recap!
We have a bunch of points on a plane, red & blue, and a linear boundary. We ask each point if it's classified correctly or where the line should go. 

The perceptron has nodes $x_{1}$ and $x_{2}$, with **edges** of the weights' values, and the 2 nodes are connected to another node with the bias. Then a decision is made as the output. Just as a human neuron kind of works, with inputs, edges/weights, a decision, and an output.

### Non-Linear Data
There are also non-linear datasets.

### Non-Linear Models
We're going to look at how to get this curve, below! We'll still use gradient descent, but it is obviously not linear.

![a curve!](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/non-linear_models.PNG)

### Neural Network Architecture
The way we will do this is to put together multiple perceptrons! Even linear ones.

![combining linear regions](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/combining_regions.PNG)

The 2 linear boundaries can be added together to get a number. Actually, they can even be weighted differently, and a bias added too! It gives a number larger than `1`, so we can just apply the **sigmoid function** to it to get the probability of a point being properly classified.

![2 linear boundaries can be added & even weighted](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/multi-layer_linear_boundary.PNG)

This looks an awful lot like a Perceptron! Yeah!

Take 2 examples and put them side by side. ![side by side](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/nn_step_1.PNG)

It's about to happen... ![neural network about to happen](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/nn_step_2.PNG)

Then combine them. A neural network!! ![neural network](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/nn_joined.PNG)

Clean them up; note that both $x_{1}$ and $x_{2}$ are still connected to both nodes by their respective weights. ![neural network all fancy-like](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/nn_cleaned_up.PNG)

And here's an alternate notation! On the left, what we're used to, the bias is in a separate node. In the right, in every layer we have a **bias unit** coming from a node with a `1` on it. You'll notice the *bias units* `-8`, `1`, and `-6`, and they all go to nodes with sigmoid activations. So the bias `-8` becomes an **edge** that goes from the bias node to the sigmoid activation node.
![alternate notation](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/nn_alternate_notation.PNG)

#### Multiple Layers
First layer is called the **input layer** and contains the inputs: $x_{1}$ and $x_{2}$

Second layer is the **hidden layer**. Set of linear models created with the input layer.

Final layer is the **output layer**, where the linear models get combined to get the non-linear model.

Obviously, there are many different models; they don't all look the same. We can add more nodes to the input, hidden, and output layers. We can also add more layers!

For example, you could have one with 2 inputs but 3 hidden layers, combining 3 linear models to get a triangular model in the output layer.  ![more hidden layers](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/extra_hidden_layers.PNG)

Or, if we have 3 inputs? We're in the 3-D space then! In general, `n` nodes means `n-D` space. ![3 inputs means 3-D space](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/triple_inputs.PNG)

Or, more output nodes? More outputs! It might be a multi-class classification model, like a 3-output of `cat`, `dog`, `bird`. ![3 outputs?](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/triple_output.PNG)

Or, more layers? This is a Deep Neural Network. Each linear model combines with others to get more and more complex models. ![more layers?](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/more_layers.PNG)  And they can just keep going, getting more & more complex! Think self-driving cars, etc. ![layers and layers and layers](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/layers_and_layers_and_layers.PNG)

#### Multi-Class Classification
How do we get multi-class outputs?  

One way is to build a model for *each* output probability. This seems like overkill.

A better way is to just have output nodes for each class, giving a probability for each animal (one node => probability of one animal), and to calculate the Softmax Function.

### Feedforward
Now that we've defined *neural networks*, we need to learn how to train them. This just means "What parameters should they have on the edges in order to model our data well?"

![feedforward diagram (partial)](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/feedforward.PNG)

**Feedforward** is the process neural networks use to turn the input into an output.  

In the image above, we have a linear boundaries $Wx + b$. No matter how many nodes we're working with, we have our `x` vector $(x_{1}, x_{2})$ and `b` which can be considered in neutral to be `1`, which gives us our first vector $\begin{bmatrix} x_1\\ x_2\\ 1\\
\end{bmatrix}$.

Next, we multiply by the matrix $W^{(1)}$ which is all the **input layer**'s weights, as shown in red in the image.

Then we take the next layer, which is the Sigmoid nodes, meaning we apply the Sigmoid function.  
![feedforward sigmoid step](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/feedforward-sigmoid-step.PNG)

Then there's another set of weights $W^{(2)}$ multiplied by the other bias `1`, and then another Sigmoid function.

All of this demonstrates the steps taken by a neural network to determine a prediction.

>![feedforward formula](https://raw.githubusercontent.com/shamicker/pytorch-challenge/master/images/feedforward_formula.PNG)

To train our *NN*, we need to formulate an Error Function. But look at that, we can use the same formula as before, even though our $\hat{y}$ is now a bit more complicated. Doesn't matter. Error formula still gives us a measure of how badly we're doing - something we still want to minimize.

$E(W) = -\frac{1}{m} \Sigma^m_{i=1} y_{i}ln(\hat{y}_{i}) + (1 - y_{i})ln(1 - \hat{y}_{i})$

### Backpropagation

### Pre-Notebook: Analyzing Student Data

### Notebook: Analyzing Student Data

### Training Optimization

### Testing

### Overfitting and Underfitting

### Early Stopping

### Regularization

### Regularization 2

### Dropout

### Local Minima

### Random Restart

### Vanishing Gradient

### Other Activation Functions

### Batch vs Stochastic Gradient Descent

### Learning Rate Decay

### Momentum

### Error Functions Around the World



## Talking PyTorch with Soumith Chintala
Hear from Soumith Chintala, the creator of PyTorch, about the past, present, and future of the PyTorch framework.

ETA 30 minutes

## Introducation to PyTorch
Learn how to use PyTorch to build and train deep neural networks. By the end of this lesson, you will build a network that can classify images of dogs and cats with state-of-the-art performance.

ETA 2 hours

## Convolutional Neural Networks
Learn how to use convolutional neural networks to build state-of-the-art computer vision models.

ETA 5 hours

## Style Transfer
Use a deep neural network to transfer the artistic style of one image onto another image.

ETA 5 hours

## Recurrent Neural Networks
Learn how to use recurrent neural networks to learn from sequential data such as text. Build a network that can generate realistic text one letter at a time.

ETA 5 hours

## Sentiment Prediction with RNNs
Here you'll build a recurrent neural network that can accurately predict the sentiment of movie reviews.

ETA 2 hours

## Deploying PyTorch Models
In this lesson, we'll walk through a tutorial showing how to deploy PyTorch models with Torch Script.

ETA 30 minutes

## Challenge Project
Build and train a model that identifies flower species from images.

__*This is part of the scholarship assessment!*__
We'll build a deep learning model from scratch!
After training and optimizing your model, it'll be uploaded to a workspace where it will receive a score based on accuracy predicting flower species from a test set. This score will be used in the decision process for awarding scholarships!!!

