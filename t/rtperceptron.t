use strict;
use warnings FATAL => qw(all);
use Program qw(lambda curry Program);
use Test::More;
use Data::Dumper;
use Carp;
use lib 'modules';
use RT::Perceptron qw(dot);
use List::Util qw(sum);

our $DEBUG=0;


sub perceptron($$$) {
  my ($input, $weight, $bias) = @_;
  print STDERR "Perceptron inputs:\n" if $DEBUG;
  use Data::Dumper;
  print Dumper([$input,$weight,$bias]) if $DEBUG;
  die "Not an array reference" unless ref($input) eq "ARRAY" and ref($weight) eq "ARRAY";
  my $v = dot($weight, $input) + $bias;
  print STDERR "Perceptron result: $v\n" if $DEBUG;
  return $v >= 0 ? 1 : 0;
};


note "Testing our dot product function";
is dot([1,3,-5],[4,-2,-1]), 3;

my $NOT_percep = lambda {
  return perceptron($_[0], [-1], 0.5);
};

my $NOTptron = lambda {
  return perceptron([$_[0]], [-1], 0.5);
};

note "Checking our NOT perceptron";
is $NOT_percep->([0]), 1;
is $NOT_percep->([1]), 0;

is $NOTptron->(0), 1;
is $NOTptron->(1), 0;

my $AND_percep = lambda{
  return perceptron($_[0], [1,1], -1.5);
};

note "Checking our AND perceptron";
is $AND_percep->([1,1]), 1; 
is $AND_percep->([0,1]), 0; 
is $AND_percep->([1,0]), 0; 
is $AND_percep->([0,0]), 0; 


my $ANDptron = lambda { $AND_percep->([@_])};

is $ANDptron->(0,0), 0;
is $ANDptron->(1,1), 1;
is $ANDptron->(0,1), 0;
is $ANDptron->(1,0), 0;

note "Checking NAND";

my $NANDptron = lambda{
  return perceptron([@_], [-1,-1], 1.5);
};

is $NANDptron->(0,0),1;
is $NANDptron->(1,1),0;
is $NANDptron->(1,0),1;
is $NANDptron->(0,1),1;

my $ORptron = lambda{
  return perceptron([@_], [1,1], -0.5);
};

note "Construction of an XOR preceptron";

my $XORptron = lambda{
  my $nandresult = $NANDptron->($_[0], $_[1]);
  my $orresult = $ORptron->($_[0], $_[1]);
  return $ANDptron->($nandresult, $orresult);
};

is $XORptron->(1,1), 0;
is $XORptron->(1,0), 1;
is $XORptron->(0,1), 1;
is $XORptron->(0,0), 0;

note "Testing XOR program integration";

my $XORprogtron = Program(
  sub {
    my $inp = shift;
    return [$NANDptron->($inp->[0], $inp->[1]), $ORptron->($inp->[0], $inp->[1])];
  },
  $AND_percep,
);

is $XORprogtron->('program')->([1,1]), 0;
is $XORprogtron->('program')->([0,0]), 0;
is $XORprogtron->('program')->([1,0]), 1;
is $XORprogtron->('program')->([0,1]), 1;



my $rtp = RT::Perceptron->new(
  w => [0.0, 0.0],
  r => 0.02,
  b => -0.1
);
$rtp->trainw([
  1 => [1,1],
  0 => [0,0],
  0 => [0,1],
  0 => [1,0]
],1);

note "Testing our own perceptron";
note explain $rtp;

is $rtp->input(0,0), 0;
is $rtp->input(1,1), 1;
is $rtp->input(0,1), 0;
is $rtp->input(1,0), 0;

my $rtp2 = RT::Perceptron->new(
  w => [0.0],
  b => -0.00000000000001,
  r => 0.02
);

$rtp2->trainw(
  [
    1 => [2],
    1 => [3],
    1 => [4],
    0 => [0],
    0 => [-1],
    0 => [-2],
    0 => [-3],
  ],
  1
);

note "Perceptron that identifies positive and negative integers";

is $rtp2->input(0), 0;
is $rtp2->input(2), 1;
is $rtp2->input(-2), 0;
is $rtp2->input(10), 1;
is $rtp2->input(-10), 0;
is $rtp2->input(-1123545), 0;
is $rtp2->input(13545), 1;
is $rtp2->input(0.01), 1;
is $rtp2->input(0.001), 1;
is $rtp2->input(0.0001), 1;
is $rtp2->input(0.00000001), 1;
is $rtp2->input(-0.0000001), 0;

my $rtp3 = RT::Perceptron->new(
  w => [0.0],
  b => 0.1,
  r => 0.4,
  t => 0.0
);
my @trainsetwords_1 = map { (1 => ["0." . join "",unpack("I*",$_)]) } qw(good nice man great);
my @trainsetwords_0 = map { (0 => ["0." . join "",unpack("I*",$_)]) } qw(ungood unnice unman ungreat);

$rtp3->trainw([
  @trainsetwords_0,
  @trainsetwords_1,
], 5);

note explain $rtp3;
note "Instead of positive/negative word classifiction, we found prefix detection";
note "Could be used to detect certain specific prefix paths like /usr/bin, etc.";
note "In the same way, suffix detection will work, can be used to detect file extensions if nothing else";
is $rtp3->input("0." . join "",unpack("I*","good")), 1;
is $rtp3->input("0." . join "",unpack("I*","ungood")), 0;
is $rtp3->input("0." . join "",unpack("I*","great")), 1;
is $rtp3->input("0." . join "",unpack("I*","ungreat")), 0;
note "Just weird but interesting results. Practical application unclear. (typo detection maybe?)";
is $rtp3->input("0." . join "",unpack("I*","nogreat")), 0;
is $rtp3->input("0." . join "",unpack("I*","xxgreat")), 0;
is $rtp3->input("0." . join "",unpack("I*","xxgrt")), 0;
is $rtp3->input("0." . join "",unpack("I*","xxgt")), 0;
is $rtp3->input("0." . join "",unpack("I*","ood")), 1;
is $rtp3->input("0." . join "",unpack("I*","unood")), 0;
is $rtp3->input("0." . join "",unpack("I*","gooder")), 1;
# @trainsetwords_1 = ();
# @trainsetwords_0 = ();
# @trainsetwords_1 = map { (1 => [("0.".unpack("%C*", $_))+0.0] ) } qw(good nice great like);
# @trainsetwords_0 = map { (0 => [("0.".unpack("%C*", $_))+0.0] ) } qw(evil bad bleh dislike);

# my $rtp4 = RT::Perceptron->new(
#   w => [1.0],
#   b => 0.1,
#   r => 0.005,
#   t => 0.1
# );
# note explain [@trainsetwords_0, @trainsetwords_1];
# $rtp4->trainw([
#   @trainsetwords_1,
#   @trainsetwords_0,
# ], 100000);


# note explain $rtp4;
# note explain \@trainsetwords_0;
# note explain \@trainsetwords_1;
# is $rtp4->input("0." . (unpack("%C*","like"))), 1;
# is $rtp4->input("0." . (unpack("%C*","dislike"))), 0;
# is $rtp4->input("0." . (unpack("%C*","good"))), 1;
# is $rtp4->input("0." . (unpack("%C*","bad"))), 0;
# is $rtp4->input("0." . (unpack("%C*","goodie"))), 1;

my $rtp5 = RT::Perceptron->new(
  w => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  b => -0.10,
  t => 0.01,
  r => 0.04
);
sub numerify_word($) {
  my @r = map { map {ord($_)/100} split //,$_} shift();
  while(@r < 8){
    push @r, 0.0;
  }
  return @r;
}
@trainsetwords_1 = map {  (1 => [numerify_word $_])  } qw(good like nice);
@trainsetwords_0 = map {  (0 => [numerify_word $_])  } qw(bad dislike mean);

$rtp5->trainw(
  [
    @trainsetwords_1,
    @trainsetwords_0
  ],
  1
);
note explain $rtp5;

note "These tests work by pure accident and shouldnt be relied on";

is $rtp5->input(numerify_word "good"), 1;
is $rtp5->input(numerify_word "bad"), 0;
is $rtp5->input(numerify_word "like"), 1;
is $rtp5->input(numerify_word "dislike"), 0;
is $rtp5->input(numerify_word "mean"), 0;
is $rtp5->input(numerify_word "nice"), 1;

done_testing;