use 5.032001;
use ExtUtils::MakeMaker;
# See lib/ExtUtils/MakeMaker.pm for details of how to influence
# the contents of the Makefile that is written.
WriteMakefile(
    NAME              => 'RT::Perceptron',
    VERSION_FROM      => 'lib/RT/Perceptron.pm', # finds $VERSION, requires EU::MM from perl >= 5.5
    PREREQ_PM         => {
        q/Test::More/ => 0,
        q/Data::Dumper/ => 0,
        q/Carp/        => 0,
        q/Exporter/     => 0
    }, # e.g., Module::Name => 1.1
    ABSTRACT_FROM     => 'lib/RT/Perceptron.pm', # retrieve abstract from module
    AUTHOR            => 'Ernest Deak <gordon.zar@gmail.com>',
    LICENSE           => 'gpl',
    #Value must be from legacy list of licenses here
    #https://metacpan.org/pod/Module::Build::API
);
