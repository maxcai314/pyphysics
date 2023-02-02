% mecanum_mpc : A fast customized optimization solver.
% 
% Copyright (C) 2013-2023 EMBOTECH AG [info@embotech.com]. All rights reserved.
% 
% 
% This software is intended for simulation and testing purposes only. 
% Use of this software for any commercial purpose is prohibited.
% 
% This program is distributed in the hope that it will be useful.
% EMBOTECH makes NO WARRANTIES with respect to the use of the software 
% without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
% PARTICULAR PURPOSE. 
% 
% EMBOTECH shall not have any liability for any damage arising from the use
% of the software.
% 
% This Agreement shall exclusively be governed by and interpreted in 
% accordance with the laws of Switzerland, excluding its principles
% of conflict of laws. The Courts of Zurich-City shall have exclusive 
% jurisdiction in case of any dispute.
% 
classdef mecanum_mpcBuildable < coder.ExternalDependency

    methods (Static)
        
        function name = getDescriptiveName(~)
            name = mfilename;
        end
        
        function b = isSupportedContext(context)
            b = context.isMatlabHostTarget();
        end
        
        function updateBuildInfo(buildInfo, cfg)
            buildablepath = fileparts(mfilename('fullpath'));
            [solverpath, foldername] = fileparts(buildablepath);
            [~, solvername] = fileparts(solverpath);
            % if the folder structure does not match to the interface folder, we assume it's the directory that contains the solver
            if(~strcmp(foldername, 'interface') || ~strcmp(solvername, 'mecanum_mpc'))
                solverpath = fullfile(buildablepath, 'mecanum_mpc');
            end
            ForcesUpdateBuildInfo(buildInfo, cfg, 'mecanum_mpc', solverpath, 0, true);
        end
        
        function [output,exitflag,info] = forcesInitOutputsMatlab()
            infos_it = coder.nullcopy(zeros(1, 1));
            infos_it2opt = coder.nullcopy(zeros(1, 1));
            infos_res_eq = coder.nullcopy(zeros(1, 1));
            infos_res_ineq = coder.nullcopy(zeros(1, 1));
            infos_rsnorm = coder.nullcopy(zeros(1, 1));
            infos_rcompnorm = coder.nullcopy(zeros(1, 1));
            infos_pobj = coder.nullcopy(zeros(1, 1));
            infos_dobj = coder.nullcopy(zeros(1, 1));
            infos_dgap = coder.nullcopy(zeros(1, 1));
            infos_rdgap = coder.nullcopy(zeros(1, 1));
            infos_mu = coder.nullcopy(zeros(1, 1));
            infos_mu_aff = coder.nullcopy(zeros(1, 1));
            infos_sigma = coder.nullcopy(zeros(1, 1));
            infos_lsit_aff = coder.nullcopy(zeros(1, 1));
            infos_lsit_cc = coder.nullcopy(zeros(1, 1));
            infos_step_aff = coder.nullcopy(zeros(1, 1));
            infos_step_cc = coder.nullcopy(zeros(1, 1));
            infos_solvetime = coder.nullcopy(zeros(1, 1));
            infos_fevalstime = coder.nullcopy(zeros(1, 1));
            infos_solver_id = coder.nullcopy(zeros(8, 1));
            info = struct('it', infos_it,...
                          'it2opt', infos_it2opt,...
                          'res_eq', infos_res_eq,...
                          'res_ineq', infos_res_ineq,...
                          'rsnorm', infos_rsnorm,...
                          'rcompnorm', infos_rcompnorm,...
                          'pobj', infos_pobj,...
                          'dobj', infos_dobj,...
                          'dgap', infos_dgap,...
                          'rdgap', infos_rdgap,...
                          'mu', infos_mu,...
                          'mu_aff', infos_mu_aff,...
                          'sigma', infos_sigma,...
                          'lsit_aff', infos_lsit_aff,...
                          'lsit_cc', infos_lsit_cc,...
                          'step_aff', infos_step_aff,...
                          'step_cc', infos_step_cc,...
                          'solvetime', infos_solvetime,...
                          'fevalstime', infos_fevalstime,...
                          'solver_id', infos_solver_id);

            outputs_x01 = coder.nullcopy(zeros(10, 1));
            outputs_x02 = coder.nullcopy(zeros(10, 1));
            outputs_x03 = coder.nullcopy(zeros(10, 1));
            outputs_x04 = coder.nullcopy(zeros(10, 1));
            outputs_x05 = coder.nullcopy(zeros(10, 1));
            outputs_x06 = coder.nullcopy(zeros(10, 1));
            outputs_x07 = coder.nullcopy(zeros(10, 1));
            outputs_x08 = coder.nullcopy(zeros(10, 1));
            outputs_x09 = coder.nullcopy(zeros(10, 1));
            outputs_x10 = coder.nullcopy(zeros(10, 1));
            output = struct('x01', outputs_x01,...
                            'x02', outputs_x02,...
                            'x03', outputs_x03,...
                            'x04', outputs_x04,...
                            'x05', outputs_x05,...
                            'x06', outputs_x06,...
                            'x07', outputs_x07,...
                            'x08', outputs_x08,...
                            'x09', outputs_x09,...
                            'x10', outputs_x10);
            
            exitflag = coder.nullcopy(0);
        end

        function [output,exitflag,info] = forcesCallWithParams(params)
            [output,exitflag,info] = mecanum_mpcBuildable.forcesCall(params.x0, params.xinit, params.all_parameters);
        end

        function [output,exitflag,info] = forcesCall(x0, xinit, all_parameters)
            solvername = 'mecanum_mpc';

            
            params = struct('x0', double(x0),...
                            'xinit', double(xinit),...
                            'all_parameters', double(all_parameters));

            [output_c, exitflag_c, info_c] = mecanum_mpcBuildable.forcesInitOutputsC();
            
            headerName = [solvername '.h'];
            coder.cinclude(headerName);
            coder.cinclude([solvername '_memory.h']);
            coder.cinclude([solvername '_adtool2forces.h']);
            % define memory pointer
            memptr = coder.opaque([solvername '_mem *'], 'HeaderFile', headerName);
            memptr = coder.ceval([solvername '_internal_mem'], uint32(0));
            % define solver input information (params, file and casadi)
            coder.cstructname(params, [solvername '_params'], 'extern', 'HeaderFile', headerName);
            fp = coder.opaque('FILE *', 'NULL', 'HeaderFile', headerName);
            % need define extern int solvername_adtool2forces(solvername_float *x, solvername_float *y, solvername_float *l, solvername_float *p, solvername_float *f, solvername_float *nabla_f, solvername_float *c, solvername_float *nabla_c, solvername_float *h, solvername_float *nabla_h, solvername_float *hess, solver_int32_default stage, solver_int32_default iteration);
            casadi = coder.opaque([solvername '_extfunc'],['&' solvername '_adtool2forces'],'HeaderFile',headerName);
            % define solver output information (output, exitflag, info)
            coder.cstructname(output_c,[solvername '_output'], 'extern', 'HeaderFile', headerName);
            coder.cstructname(info_c,[solvername '_info'], 'extern', 'HeaderFile', headerName);
            exitflag_c = coder.ceval([solvername '_solve'], coder.rref(params), ...
                                      coder.wref(output_c), coder.wref(info_c), ... 
                                      memptr, fp, casadi);
            
            [output, exitflag, info] = mecanum_mpcBuildable.forcesInitOutputsMatlab();

            info.it = cast(info_c.it, 'like', info.it);
            info.it2opt = cast(info_c.it2opt, 'like', info.it2opt);
            info.res_eq = cast(info_c.res_eq, 'like', info.res_eq);
            info.res_ineq = cast(info_c.res_ineq, 'like', info.res_ineq);
            info.rsnorm = cast(info_c.rsnorm, 'like', info.rsnorm);
            info.rcompnorm = cast(info_c.rcompnorm, 'like', info.rcompnorm);
            info.pobj = cast(info_c.pobj, 'like', info.pobj);
            info.dobj = cast(info_c.dobj, 'like', info.dobj);
            info.dgap = cast(info_c.dgap, 'like', info.dgap);
            info.rdgap = cast(info_c.rdgap, 'like', info.rdgap);
            info.mu = cast(info_c.mu, 'like', info.mu);
            info.mu_aff = cast(info_c.mu_aff, 'like', info.mu_aff);
            info.sigma = cast(info_c.sigma, 'like', info.sigma);
            info.lsit_aff = cast(info_c.lsit_aff, 'like', info.lsit_aff);
            info.lsit_cc = cast(info_c.lsit_cc, 'like', info.lsit_cc);
            info.step_aff = cast(info_c.step_aff, 'like', info.step_aff);
            info.step_cc = cast(info_c.step_cc, 'like', info.step_cc);
            info.solvetime = cast(info_c.solvetime, 'like', info.solvetime);
            info.fevalstime = cast(info_c.fevalstime, 'like', info.fevalstime);
            info.solver_id = cast(info_c.solver_id, 'like', info.solver_id);

            output.x01 = cast(output_c.x01, 'like', output.x01);
            output.x02 = cast(output_c.x02, 'like', output.x02);
            output.x03 = cast(output_c.x03, 'like', output.x03);
            output.x04 = cast(output_c.x04, 'like', output.x04);
            output.x05 = cast(output_c.x05, 'like', output.x05);
            output.x06 = cast(output_c.x06, 'like', output.x06);
            output.x07 = cast(output_c.x07, 'like', output.x07);
            output.x08 = cast(output_c.x08, 'like', output.x08);
            output.x09 = cast(output_c.x09, 'like', output.x09);
            output.x10 = cast(output_c.x10, 'like', output.x10);
            
            exitflag = exitflag_c;
        end
    end

    methods (Static, Access = private)
        function [output,exitflag,info] = forcesInitOutputsC()
            infos_it = coder.nullcopy(int32(zeros(1, 1)));
            infos_it2opt = coder.nullcopy(int32(zeros(1, 1)));
            infos_res_eq = coder.nullcopy(double(zeros(1, 1)));
            infos_res_ineq = coder.nullcopy(double(zeros(1, 1)));
            infos_rsnorm = coder.nullcopy(double(zeros(1, 1)));
            infos_rcompnorm = coder.nullcopy(double(zeros(1, 1)));
            infos_pobj = coder.nullcopy(double(zeros(1, 1)));
            infos_dobj = coder.nullcopy(double(zeros(1, 1)));
            infos_dgap = coder.nullcopy(double(zeros(1, 1)));
            infos_rdgap = coder.nullcopy(double(zeros(1, 1)));
            infos_mu = coder.nullcopy(double(zeros(1, 1)));
            infos_mu_aff = coder.nullcopy(double(zeros(1, 1)));
            infos_sigma = coder.nullcopy(double(zeros(1, 1)));
            infos_lsit_aff = coder.nullcopy(int32(zeros(1, 1)));
            infos_lsit_cc = coder.nullcopy(int32(zeros(1, 1)));
            infos_step_aff = coder.nullcopy(double(zeros(1, 1)));
            infos_step_cc = coder.nullcopy(double(zeros(1, 1)));
            infos_solvetime = coder.nullcopy(double(zeros(1, 1)));
            infos_fevalstime = coder.nullcopy(double(zeros(1, 1)));
            infos_solver_id = coder.nullcopy(int32(zeros(8, 1)));
            info = struct('it', infos_it,...
                          'it2opt', infos_it2opt,...
                          'res_eq', infos_res_eq,...
                          'res_ineq', infos_res_ineq,...
                          'rsnorm', infos_rsnorm,...
                          'rcompnorm', infos_rcompnorm,...
                          'pobj', infos_pobj,...
                          'dobj', infos_dobj,...
                          'dgap', infos_dgap,...
                          'rdgap', infos_rdgap,...
                          'mu', infos_mu,...
                          'mu_aff', infos_mu_aff,...
                          'sigma', infos_sigma,...
                          'lsit_aff', infos_lsit_aff,...
                          'lsit_cc', infos_lsit_cc,...
                          'step_aff', infos_step_aff,...
                          'step_cc', infos_step_cc,...
                          'solvetime', infos_solvetime,...
                          'fevalstime', infos_fevalstime,...
                          'solver_id', infos_solver_id);
                          
            outputs_x01 = coder.nullcopy(double(zeros(10, 1)));
            outputs_x02 = coder.nullcopy(double(zeros(10, 1)));
            outputs_x03 = coder.nullcopy(double(zeros(10, 1)));
            outputs_x04 = coder.nullcopy(double(zeros(10, 1)));
            outputs_x05 = coder.nullcopy(double(zeros(10, 1)));
            outputs_x06 = coder.nullcopy(double(zeros(10, 1)));
            outputs_x07 = coder.nullcopy(double(zeros(10, 1)));
            outputs_x08 = coder.nullcopy(double(zeros(10, 1)));
            outputs_x09 = coder.nullcopy(double(zeros(10, 1)));
            outputs_x10 = coder.nullcopy(double(zeros(10, 1)));
            output = struct('x01', outputs_x01,...
                            'x02', outputs_x02,...
                            'x03', outputs_x03,...
                            'x04', outputs_x04,...
                            'x05', outputs_x05,...
                            'x06', outputs_x06,...
                            'x07', outputs_x07,...
                            'x08', outputs_x08,...
                            'x09', outputs_x09,...
                            'x10', outputs_x10);
            exitflag = coder.nullcopy(int32(0));
        end
    end

    
end
