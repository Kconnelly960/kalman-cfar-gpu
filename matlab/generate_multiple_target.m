%% Range-Speed Response Pattern of Target
% This example shows how to visualize the speed and range of a target in a
% pulsed radar system that uses a rectangular waveform.
%%
% Place an isotropic antenna element at the global origin
% _(0,0,0)_. Then, place a target with a nonfluctuating
% RCS of 1 square meter at _(5000,5000,10)_, which is
% approximately 7 km from the transmitter. Set the operating (carrier)
% frequency to 10 GHz. To simulate a monostatic radar, set the
% |InUseOutputPort| property on the transmitter to
% |true|. Calculate the range and angle from the
% transmitter to the target.

%%
%

antenna = phased.IsotropicAntennaElement('FrequencyRange',[5e9 15e9]);
transmitter = phased.Transmitter('Gain',20,'InUseOutputPort',true);
fc = 10e9;

% Define three targets
target1 = phased.RadarTarget('Model','Nonfluctuating','MeanRCS',1,'OperatingFrequency',fc);
target2 = phased.RadarTarget('Model','Nonfluctuating','MeanRCS',1.5,'OperatingFrequency',fc);
target3 = phased.RadarTarget('Model','Nonfluctuating','MeanRCS',0.5,'OperatingFrequency',fc);

% Define locations of each target
tgtloc1 = [5000; 0; 0];
tgtloc2 = [6000; 500; 0];
tgtloc3 = [4500; -500; 0];

txloc = [0; 0; 0];
antennaplatform = phased.Platform('InitialPosition',txloc);
targetplatform1 = phased.Platform('InitialPosition',tgtloc1);
targetplatform2 = phased.Platform('InitialPosition',tgtloc2);
targetplatform3 = phased.Platform('InitialPosition',tgtloc3);


% Waveform and channel
waveform = phased.RectangularWaveform('PulseWidth',2e-6, 'OutputFormat','Pulses','PRF',1e4,'NumPulses',1000);
c = physconst('LightSpeed');
maxrange = c / (2 * waveform.PRF);
lambda = c / fc;
SNR = npwgnthresh(1e-6,1,'noncoherent');
dbterm = db2pow(SNR - 2 * transmitter.Gain);
transmitter.PeakPower = (4 * pi)^3 * physconst('Boltzmann') * 290 / waveform.PulseWidth / lambda^2 * maxrange^4 * dbterm;

radiator = phased.Radiator('PropagationSpeed',c, 'OperatingFrequency',fc,'Sensor',antenna);
channel = phased.FreeSpace('PropagationSpeed',c, 'OperatingFrequency',fc,'TwoWayPropagation',false);
collector = phased.Collector('PropagationSpeed',c, 'OperatingFrequency',fc,'Sensor',antenna);
receiver = phased.ReceiverPreamp('NoiseFigure',0, 'EnableInputPort',true,'SeedSource','Property','Seed',2e3);

% Simulation
numPulses = 25;
rx_puls = zeros(100000,numPulses);
% Simulation loop
for n = 1:numPulses
    wf = waveform();
    [wf, txstatus] = transmitter(wf);
    
    % Radiate pulse toward each target individually and collect echoes
    wf1 = radiator(wf, tgtang1);
    wf1 = channel(wf1, txloc, tgtloc1, [0;0;0], [0;0;0]);
    wf1 = target1(wf1);
    wf1 = channel(wf1, tgtloc1, txloc, [0;0;0], [0;0;0]);
    wf1 = collector(wf1, tgtang1);
    
    wf2 = radiator(wf, tgtang2);
    wf2 = channel(wf2, txloc, tgtloc2, [0;0;0], [0;0;0]);
    wf2 = target2(wf2);
    wf2 = channel(wf2, tgtloc2, txloc, [0;0;0], [0;0;0]);
    wf2 = collector(wf2, tgtang2);
    
    wf3 = radiator(wf, tgtang3);
    wf3 = channel(wf3, txloc, tgtloc3, [0;0;0], [0;0;0]);
    wf3 = target3(wf3);
    wf3 = channel(wf3, tgtloc3, txloc, [0;0;0], [0;0;0]);
    wf3 = collector(wf3, tgtang3);

    % Combine the echoes from all targets
    wf = wf1 + wf2 + wf3;

    % Receive the combined echoes
    rx_puls(:, n) = receiver(wf, ~txstatus);
end

rangedoppler = phased.RangeDopplerResponse('RangeMethod','Matched Filter','PropagationSpeed',c,'DopplerOutput','Speed','OperatingFrequency',fc);
csvwrite('radarDataMatlab_MultipleTarget.csv', rx_puls);




%plotResponse(rangedoppler,rx_puls,getMatchedFilter(waveform))
%%
% The plot shows the stationary target at a range of approximately 7000 m.

%%
% Copyright 2012 The MathWorks, Inc.