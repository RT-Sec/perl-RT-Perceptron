# RT::Perceptron

Custom implementation of the Perceptron algorithm.  
You can check how this is implemented by checking out  
my document file on it:  
https://www.patreon.com/posts/machine-learning-57940147  
  
Alternatively you can study it yourself here:  
https://en.wikipedia.org/wiki/Perceptron  

Documentation is provided with `perldoc RT::Perceptron`  
after installation.

## Simple usage

```perl
  use RT::Perceptron;
  use Test::More;

  my $rtp = RT::Perceptron->new(
    w => [0.0,0.0],   #weights
    b => 0.1,         #bias
    t => 0.0,         #threshold
    r => 0.04         #learning rate
  );
  # AND gate
  $rtp->trainw([
    1 => [1,1],
    0 => [0,0],
    0 => [0,1],
    0 => [1,0]
  ],1); # MAX 1 iteration

  # Check results
  is $rtp->input(0,0), 0;
  is $rtp->input(1,1), 1;
  is $rtp->input(0,1), 0;
  is $rtp->input(1,0), 0;

  done_testing;
```

## LICENSE
See COPYING for Licensing information

