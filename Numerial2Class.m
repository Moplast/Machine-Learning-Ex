function [ y, yMap ] = Numerial2Class( Y )
% Transfer Y(vector) numerial to class
% and return a mapping vector from Y to y, 
% yMap[$transfered class$] is original numerial


%% Test

y_length = length(Y);
y = zeros(y_length, 1);
yMap = [];


if y_length == 0
	return
end

for i = 1 : y_length	
	if isempty(find(yMap == Y(i)))
		yMap = [yMap; Y(i)];
	end
	
end

yMap = sort(yMap);

for i = 1 : y_length
	findIndex = find(yMap == Y(i));
	y(i) = findIndex;	
end

y = y(:);


% % Initialization
% y_length = length(Y);
% y = zeros(y_length, 1);
% yMap = [];
% 
% if y_length == 0
% 	return
% end
% 
% mapIndex = 0;
% 
% % First value
% mapIndex = 1;		
% yMap(mapIndex) = Y(1);
% y(1) = mapIndex;
% 
% for i = 2 : y_length
% 	findIndex = find(yMap == Y(i));
% 	if isempty(findIndex)		
% 		% New numerial value
% 		mapIndex = mapIndex + 1;
% 		yMap(mapIndex) = Y(i);
% 		y(i) = mapIndex;
% 	else
% 		% has existed in yMap
% 		y(i) = findIndex;
% 	end	
% end
% 
% y = y(:);


end