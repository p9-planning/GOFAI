%% -*- prolog -*-
%%=============================================================================
%% Copyright (C) 2011 by Denys Duchier
%%
%% This program is free software: you can redistribute it and/or modify it
%% under the terms of the GNU Lesser General Public License as published by the
%% Free Software Foundation, either version 3 of the License, or (at your
%% option) any later version.
%%
%% This program is distributed in the hope that it will be useful, but WITHOUT
%% ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
%% FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
%% more details.
%%
%% You should have received a copy of the GNU Lesser General Public License
%% along with this program.  If not, see <http://www.gnu.org/licenses/>.
%%=============================================================================

is_RestartMode_('RM_NONE').
is_RestartMode_('RM_CONSTANT').
is_RestartMode_('RM_LINEAR').
is_RestartMode_('RM_LUBY').
is_RestartMode_('RM_GEOMETRIC').


is_RestartMode_('RM_NONE','RM_NONE').
is_RestartMode_('RM_CONSTANT','RM_CONSTANT').
is_RestartMode_('RM_LINEAR','RM_LINEAR').
is_RestartMode_('RM_LUBY','RM_LUBY').
is_RestartMode_('RM_GEOMETRIC','RM_GEOMETRIC').


is_RestartMode(X,Y) :- nonvar(X), is_RestartMode_(X,Y).
is_RestartMode(X) :- is_RestartMode_(X,_).


is_FloatRelType_('FRT_EQ').
is_FloatRelType_('FRT_NQ').
is_FloatRelType_('FRT_LQ').
is_FloatRelType_('FRT_LE').
is_FloatRelType_('FRT_GQ').
is_FloatRelType_('FRT_GR').


is_FloatRelType_('FRT_EQ','FRT_EQ').
is_FloatRelType_('FRT_NQ','FRT_NQ').
is_FloatRelType_('FRT_LQ','FRT_LQ').
is_FloatRelType_('FRT_LE','FRT_LE').
is_FloatRelType_('FRT_GQ','FRT_GQ').
is_FloatRelType_('FRT_GR','FRT_GR').


is_FloatRelType(X,Y) :- nonvar(X), is_FloatRelType_(X,Y).
is_FloatRelType(X) :- is_FloatRelType_(X,_).


is_ReifyMode_('RM_EQV').
is_ReifyMode_('RM_IMP').
is_ReifyMode_('RM_PMI').


is_ReifyMode_('RM_EQV','RM_EQV').
is_ReifyMode_('RM_IMP','RM_IMP').
is_ReifyMode_('RM_PMI','RM_PMI').


is_ReifyMode(X,Y) :- nonvar(X), is_ReifyMode_(X,Y).
is_ReifyMode(X) :- is_ReifyMode_(X,_).


is_IntRelType_('IRT_EQ').
is_IntRelType_('IRT_NQ').
is_IntRelType_('IRT_LQ').
is_IntRelType_('IRT_LE').
is_IntRelType_('IRT_GQ').
is_IntRelType_('IRT_GR').


is_IntRelType_('IRT_EQ','IRT_EQ').
is_IntRelType_('IRT_NQ','IRT_NQ').
is_IntRelType_('IRT_LQ','IRT_LQ').
is_IntRelType_('IRT_LE','IRT_LE').
is_IntRelType_('IRT_GQ','IRT_GQ').
is_IntRelType_('IRT_GR','IRT_GR').


is_IntRelType(X,Y) :- nonvar(X), is_IntRelType_(X,Y).
is_IntRelType(X) :- is_IntRelType_(X,_).


is_BoolOpType_('BOT_AND').
is_BoolOpType_('BOT_OR').
is_BoolOpType_('BOT_IMP').
is_BoolOpType_('BOT_EQV').
is_BoolOpType_('BOT_XOR').


is_BoolOpType_('BOT_AND','BOT_AND').
is_BoolOpType_('BOT_OR','BOT_OR').
is_BoolOpType_('BOT_IMP','BOT_IMP').
is_BoolOpType_('BOT_EQV','BOT_EQV').
is_BoolOpType_('BOT_XOR','BOT_XOR').


is_BoolOpType(X,Y) :- nonvar(X), is_BoolOpType_(X,Y).
is_BoolOpType(X) :- is_BoolOpType_(X,_).


is_IntPropLevel_('IPL_DEF').
is_IntPropLevel_('IPL_VAL').
is_IntPropLevel_('IPL_BND').
is_IntPropLevel_('IPL_DOM').
is_IntPropLevel_('IPL_BASIC').
is_IntPropLevel_('IPL_ADVANCED').
is_IntPropLevel_('IPL_BASIC_ADVANCED').
is_IntPropLevel_('_IPL_BITS').


is_IntPropLevel_('IPL_DEF','IPL_DEF').
is_IntPropLevel_('IPL_VAL','IPL_VAL').
is_IntPropLevel_('IPL_BND','IPL_BND').
is_IntPropLevel_('IPL_DOM','IPL_DOM').
is_IntPropLevel_('IPL_BASIC','IPL_BASIC').
is_IntPropLevel_('IPL_ADVANCED','IPL_ADVANCED').
is_IntPropLevel_('IPL_BASIC_ADVANCED','IPL_BASIC_ADVANCED').
is_IntPropLevel_('_IPL_BITS','_IPL_BITS').


is_IntPropLevel(X,Y) :- nonvar(X), is_IntPropLevel_(X,Y).
is_IntPropLevel(X) :- is_IntPropLevel_(X,_).


is_TaskType_('TT_FIXP').
is_TaskType_('TT_FIXS').
is_TaskType_('TT_FIXE').


is_TaskType_('TT_FIXP','TT_FIXP').
is_TaskType_('TT_FIXS','TT_FIXS').
is_TaskType_('TT_FIXE','TT_FIXE').


is_TaskType(X,Y) :- nonvar(X), is_TaskType_(X,Y).
is_TaskType(X) :- is_TaskType_(X,_).


is_TraceEvent_('TE_INIT').
is_TraceEvent_('TE_PRUNE').
is_TraceEvent_('TE_FIX').
is_TraceEvent_('TE_FAIL').
is_TraceEvent_('TE_DONE').
is_TraceEvent_('TE_PROPAGATE').
is_TraceEvent_('TE_COMMIT').
is_TraceEvent_('TE_POST').


is_TraceEvent_('TE_INIT','TE_INIT').
is_TraceEvent_('TE_PRUNE','TE_PRUNE').
is_TraceEvent_('TE_FIX','TE_FIX').
is_TraceEvent_('TE_FAIL','TE_FAIL').
is_TraceEvent_('TE_DONE','TE_DONE').
is_TraceEvent_('TE_PROPAGATE','TE_PROPAGATE').
is_TraceEvent_('TE_COMMIT','TE_COMMIT').
is_TraceEvent_('TE_POST','TE_POST').


is_TraceEvent(X,Y) :- nonvar(X), is_TraceEvent_(X,Y).
is_TraceEvent(X) :- is_TraceEvent_(X,_).


is_SetRelType_('SRT_EQ').
is_SetRelType_('SRT_NQ').
is_SetRelType_('SRT_SUB').
is_SetRelType_('SRT_SUP').
is_SetRelType_('SRT_DISJ').
is_SetRelType_('SRT_CMPL').
is_SetRelType_('SRT_LQ').
is_SetRelType_('SRT_LE').
is_SetRelType_('SRT_GQ').
is_SetRelType_('SRT_GR').


is_SetRelType_('SRT_EQ','SRT_EQ').
is_SetRelType_('SRT_NQ','SRT_NQ').
is_SetRelType_('SRT_SUB','SRT_SUB').
is_SetRelType_('SRT_SUP','SRT_SUP').
is_SetRelType_('SRT_DISJ','SRT_DISJ').
is_SetRelType_('SRT_CMPL','SRT_CMPL').
is_SetRelType_('SRT_LQ','SRT_LQ').
is_SetRelType_('SRT_LE','SRT_LE').
is_SetRelType_('SRT_GQ','SRT_GQ').
is_SetRelType_('SRT_GR','SRT_GR').


is_SetRelType(X,Y) :- nonvar(X), is_SetRelType_(X,Y).
is_SetRelType(X) :- is_SetRelType_(X,_).


is_SetOpType_('SOT_UNION').
is_SetOpType_('SOT_DUNION').
is_SetOpType_('SOT_INTER').
is_SetOpType_('SOT_MINUS').


is_SetOpType_('SOT_UNION','SOT_UNION').
is_SetOpType_('SOT_DUNION','SOT_DUNION').
is_SetOpType_('SOT_INTER','SOT_INTER').
is_SetOpType_('SOT_MINUS','SOT_MINUS').


is_SetOpType(X,Y) :- nonvar(X), is_SetOpType_(X,Y).
is_SetOpType(X) :- is_SetOpType_(X,_).


dom(X0,X1) :-
        (is_IntVar(X0,Y0)
         -> (is_IntSet(X1,Y1)
             -> gecode_constraint_dom_1(Y0,Y1)
             ;  (is_int(X1,Y1)
                 -> gecode_constraint_dom_2(Y0,Y1)
                 ;  throw(error(type_error(int(X1)),gecode_argument_error(dom(X0,X1),arg=2)))))
         ;  throw(error(type_error('IntVar'(X0)),gecode_argument_error(dom(X0,X1),arg=1)))).

dom(X0,X1,X2) :-
        (is_IntVar(X0,Y0)
         -> (is_int(X1,Y1)
             -> (is_int(X2,Y2)
                 -> gecode_constraint_dom_3(Y0,Y1,Y2)
                 ;  throw(error(type_error(int(X2)),gecode_argument_error(dom(X0,X1,X2),arg=3))))
             ;  throw(error(type_error(int(X1)),gecode_argument_error(dom(X0,X1,X2),arg=2))))
         ;  (is_Space_or_Clause(X0,Y0)
             -> (is_BoolVar(X1,Y1)
                 -> (is_BoolVar(X2,Y2)
                     -> gecode_constraint_dom_138(Y0,Y1,Y2)
                     ;  throw(error(type_error('BoolVar'(X2)),gecode_argument_error(dom(X0,X1,X2),arg=3))))
                 ;  (is_BoolVarArgs(X1,Y1)
                     -> (is_BoolVarArgs(X2,Y2)
                         -> gecode_constraint_dom_139(Y0,Y1,Y2)
                         ;  throw(error(type_error('BoolVarArgs'(X2)),gecode_argument_error(dom(X0,X1,X2),arg=3))))
                     ;  (is_FloatVarArgs(X1,Y1)
                         -> (is_FloatVarArgs(X2,Y2)
                             -> gecode_constraint_dom_140(Y0,Y1,Y2)
                             ;  (is_FloatVal(X2,Y2)
                                 -> gecode_constraint_dom_142(Y0,Y1,Y2)
                                 ;  throw(error(type_error('FloatVal'(X2)),gecode_argument_error(dom(X0,X1,X2),arg=3)))))
                         ;  (is_IntVarArgs(X1,Y1)
                             -> (is_IntSet(X2,Y2)
                                 -> gecode_constraint_dom_143(Y0,Y1,Y2)
                                 ;  (is_IntVarArgs(X2,Y2)
                                     -> gecode_constraint_dom_144(Y0,Y1,Y2)
                                     ;  (is_int(X2,Y2)
                                         -> gecode_constraint_dom_146(Y0,Y1,Y2)
                                         ;  throw(error(type_error(int(X2)),gecode_argument_error(dom(X0,X1,X2),arg=3))))))
                             ;  (is_SetVarArgs(X1,Y1)
                                 -> (is_SetVarArgs(X2,Y2)
                                     -> gecode_constraint_dom_147(Y0,Y1,Y2)
                                     ;  throw(error(type_error('SetVarArgs'(X2)),gecode_argument_error(dom(X0,X1,X2),arg=3))))
                                 ;  (is_FloatVar(X1,Y1)
                                     -> (is_FloatVal(X2,Y2)
                                         -> gecode_constraint_dom_153(Y0,Y1,Y2)
                                         ;  (is_FloatVar(X2,Y2)
                                             -> gecode_constraint_dom_155(Y0,Y1,Y2)
                                             ;  throw(error(type_error('FloatVar'(X2)),gecode_argument_error(dom(X0,X1,X2),arg=3)))))
                                     ;  (is_IntVar(X1,Y1)
                                         -> (is_IntSet(X2,Y2)
                                             -> gecode_constraint_dom_156(Y0,Y1,Y2)
                                             ;  (is_int(X2,Y2)
                                                 -> gecode_constraint_dom_159(Y0,Y1,Y2)
                                                 ;  (is_IntVar(X2,Y2)
                                                     -> gecode_constraint_dom_162(Y0,Y1,Y2)
                                                     ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(dom(X0,X1,X2),arg=3))))))
                                         ;  (is_SetVar(X1,Y1)
                                             -> (is_SetVar(X2,Y2)
                                                 -> gecode_constraint_dom_169(Y0,Y1,Y2)
                                                 ;  throw(error(type_error('SetVar'(X2)),gecode_argument_error(dom(X0,X1,X2),arg=3))))
                                             ;  throw(error(type_error('SetVar'(X1)),gecode_argument_error(dom(X0,X1,X2),arg=2)))))))))))
             ;  throw(error(type_error('Space'(X0)),gecode_argument_error(dom(X0,X1,X2),arg=1))))).

binpacking(X0,X1,X2,X3,X4,X5) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_int(X1,Y1)
             -> (is_IntVarArgs(X2,Y2)
                 -> (is_IntVarArgs(X3,Y3)
                     -> (is_IntArgs(X4,Y4)
                         -> (is_IntArgs(X5,Y5)
                             -> gecode_constraint_binpacking_4(Y0,Y1,Y2,Y3,Y4,Y5)
                             ;  throw(error(type_error('IntArgs'(X5)),gecode_argument_error(binpacking(X0,X1,X2,X3,X4,X5),arg=6))))
                         ;  throw(error(type_error('IntArgs'(X4)),gecode_argument_error(binpacking(X0,X1,X2,X3,X4,X5),arg=5))))
                     ;  throw(error(type_error('IntVarArgs'(X3)),gecode_argument_error(binpacking(X0,X1,X2,X3,X4,X5),arg=4))))
                 ;  throw(error(type_error('IntVarArgs'(X2)),gecode_argument_error(binpacking(X0,X1,X2,X3,X4,X5),arg=3))))
             ;  throw(error(type_error(int(X1)),gecode_argument_error(binpacking(X0,X1,X2,X3,X4,X5),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(binpacking(X0,X1,X2,X3,X4,X5),arg=1)))).

max(X0) :-
        (is_IntVarArgs(X0,Y0)
         -> gecode_constraint_max_5(Y0)
         ;  throw(error(type_error('IntVarArgs'(X0)),gecode_argument_error(max(X0),arg=1)))).

min(X0) :-
        (is_IntVarArgs(X0,Y0)
         -> gecode_constraint_min_6(Y0)
         ;  throw(error(type_error('IntVarArgs'(X0)),gecode_argument_error(min(X0),arg=1)))).

sum(X0) :-
        (is_BoolVarArgs(X0,Y0)
         -> gecode_constraint_sum_7(Y0)
         ;  (is_IntArgs(X0,Y0)
             -> gecode_constraint_sum_8(Y0)
             ;  (is_IntVarArgs(X0,Y0)
                 -> gecode_constraint_sum_11(Y0)
                 ;  throw(error(type_error('IntVarArgs'(X0)),gecode_argument_error(sum(X0),arg=1)))))).

sum(X0,X1) :-
        (is_IntArgs(X0,Y0)
         -> (is_BoolVarArgs(X1,Y1)
             -> gecode_constraint_sum_9(Y0,Y1)
             ;  (is_IntVarArgs(X1,Y1)
                 -> gecode_constraint_sum_10(Y0,Y1)
                 ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(sum(X0,X1),arg=2)))))
         ;  throw(error(type_error('IntArgs'(X0)),gecode_argument_error(sum(X0,X1),arg=1)))).

abs(X0,X1,X2) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_FloatVar(X1,Y1)
             -> (is_FloatVar(X2,Y2)
                 -> gecode_constraint_abs_12(Y0,Y1,Y2)
                 ;  throw(error(type_error('FloatVar'(X2)),gecode_argument_error(abs(X0,X1,X2),arg=3))))
             ;  (is_IntVar(X1,Y1)
                 -> (is_IntVar(X2,Y2)
                     -> gecode_constraint_abs_13(Y0,Y1,Y2)
                     ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(abs(X0,X1,X2),arg=3))))
                 ;  throw(error(type_error('IntVar'(X1)),gecode_argument_error(abs(X0,X1,X2),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(abs(X0,X1,X2),arg=1)))).

argmax(X0,X1,X2,X3) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVarArgs(X1,Y1)
             -> (is_int(X2,Y2)
                 -> (is_IntVar(X3,Y3)
                     -> gecode_constraint_argmax_14(Y0,Y1,Y2,Y3)
                     ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(argmax(X0,X1,X2,X3),arg=4))))
                 ;  (is_IntVar(X2,Y2)
                     -> (is_bool(X3,Y3)
                         -> gecode_constraint_argmax_17(Y0,Y1,Y2,Y3)
                         ;  throw(error(type_error(bool(X3)),gecode_argument_error(argmax(X0,X1,X2,X3),arg=4))))
                     ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(argmax(X0,X1,X2,X3),arg=3)))))
             ;  (is_IntVarArgs(X1,Y1)
                 -> (is_int(X2,Y2)
                     -> (is_IntVar(X3,Y3)
                         -> gecode_constraint_argmax_18(Y0,Y1,Y2,Y3)
                         ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(argmax(X0,X1,X2,X3),arg=4))))
                     ;  (is_IntVar(X2,Y2)
                         -> (is_bool(X3,Y3)
                             -> gecode_constraint_argmax_21(Y0,Y1,Y2,Y3)
                             ;  throw(error(type_error(bool(X3)),gecode_argument_error(argmax(X0,X1,X2,X3),arg=4))))
                         ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(argmax(X0,X1,X2,X3),arg=3)))))
                 ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(argmax(X0,X1,X2,X3),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(argmax(X0,X1,X2,X3),arg=1)))).

argmax(X0,X1,X2,X3,X4) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVarArgs(X1,Y1)
             -> (is_int(X2,Y2)
                 -> (is_IntVar(X3,Y3)
                     -> (is_bool(X4,Y4)
                         -> gecode_constraint_argmax_15(Y0,Y1,Y2,Y3,Y4)
                         ;  throw(error(type_error(bool(X4)),gecode_argument_error(argmax(X0,X1,X2,X3,X4),arg=5))))
                     ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(argmax(X0,X1,X2,X3,X4),arg=4))))
                 ;  throw(error(type_error(int(X2)),gecode_argument_error(argmax(X0,X1,X2,X3,X4),arg=3))))
             ;  (is_IntVarArgs(X1,Y1)
                 -> (is_int(X2,Y2)
                     -> (is_IntVar(X3,Y3)
                         -> (is_bool(X4,Y4)
                             -> gecode_constraint_argmax_19(Y0,Y1,Y2,Y3,Y4)
                             ;  throw(error(type_error(bool(X4)),gecode_argument_error(argmax(X0,X1,X2,X3,X4),arg=5))))
                         ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(argmax(X0,X1,X2,X3,X4),arg=4))))
                     ;  throw(error(type_error(int(X2)),gecode_argument_error(argmax(X0,X1,X2,X3,X4),arg=3))))
                 ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(argmax(X0,X1,X2,X3,X4),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(argmax(X0,X1,X2,X3,X4),arg=1)))).

argmax(X0,X1,X2) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVarArgs(X1,Y1)
             -> (is_IntVar(X2,Y2)
                 -> gecode_constraint_argmax_16(Y0,Y1,Y2)
                 ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(argmax(X0,X1,X2),arg=3))))
             ;  (is_IntVarArgs(X1,Y1)
                 -> (is_IntVar(X2,Y2)
                     -> gecode_constraint_argmax_20(Y0,Y1,Y2)
                     ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(argmax(X0,X1,X2),arg=3))))
                 ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(argmax(X0,X1,X2),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(argmax(X0,X1,X2),arg=1)))).

argmin(X0,X1,X2,X3) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVarArgs(X1,Y1)
             -> (is_int(X2,Y2)
                 -> (is_IntVar(X3,Y3)
                     -> gecode_constraint_argmin_22(Y0,Y1,Y2,Y3)
                     ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(argmin(X0,X1,X2,X3),arg=4))))
                 ;  (is_IntVar(X2,Y2)
                     -> (is_bool(X3,Y3)
                         -> gecode_constraint_argmin_25(Y0,Y1,Y2,Y3)
                         ;  throw(error(type_error(bool(X3)),gecode_argument_error(argmin(X0,X1,X2,X3),arg=4))))
                     ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(argmin(X0,X1,X2,X3),arg=3)))))
             ;  (is_IntVarArgs(X1,Y1)
                 -> (is_int(X2,Y2)
                     -> (is_IntVar(X3,Y3)
                         -> gecode_constraint_argmin_26(Y0,Y1,Y2,Y3)
                         ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(argmin(X0,X1,X2,X3),arg=4))))
                     ;  (is_IntVar(X2,Y2)
                         -> (is_bool(X3,Y3)
                             -> gecode_constraint_argmin_29(Y0,Y1,Y2,Y3)
                             ;  throw(error(type_error(bool(X3)),gecode_argument_error(argmin(X0,X1,X2,X3),arg=4))))
                         ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(argmin(X0,X1,X2,X3),arg=3)))))
                 ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(argmin(X0,X1,X2,X3),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(argmin(X0,X1,X2,X3),arg=1)))).

argmin(X0,X1,X2,X3,X4) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVarArgs(X1,Y1)
             -> (is_int(X2,Y2)
                 -> (is_IntVar(X3,Y3)
                     -> (is_bool(X4,Y4)
                         -> gecode_constraint_argmin_23(Y0,Y1,Y2,Y3,Y4)
                         ;  throw(error(type_error(bool(X4)),gecode_argument_error(argmin(X0,X1,X2,X3,X4),arg=5))))
                     ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(argmin(X0,X1,X2,X3,X4),arg=4))))
                 ;  throw(error(type_error(int(X2)),gecode_argument_error(argmin(X0,X1,X2,X3,X4),arg=3))))
             ;  (is_IntVarArgs(X1,Y1)
                 -> (is_int(X2,Y2)
                     -> (is_IntVar(X3,Y3)
                         -> (is_bool(X4,Y4)
                             -> gecode_constraint_argmin_27(Y0,Y1,Y2,Y3,Y4)
                             ;  throw(error(type_error(bool(X4)),gecode_argument_error(argmin(X0,X1,X2,X3,X4),arg=5))))
                         ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(argmin(X0,X1,X2,X3,X4),arg=4))))
                     ;  throw(error(type_error(int(X2)),gecode_argument_error(argmin(X0,X1,X2,X3,X4),arg=3))))
                 ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(argmin(X0,X1,X2,X3,X4),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(argmin(X0,X1,X2,X3,X4),arg=1)))).

argmin(X0,X1,X2) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVarArgs(X1,Y1)
             -> (is_IntVar(X2,Y2)
                 -> gecode_constraint_argmin_24(Y0,Y1,Y2)
                 ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(argmin(X0,X1,X2),arg=3))))
             ;  (is_IntVarArgs(X1,Y1)
                 -> (is_IntVar(X2,Y2)
                     -> gecode_constraint_argmin_28(Y0,Y1,Y2)
                     ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(argmin(X0,X1,X2),arg=3))))
                 ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(argmin(X0,X1,X2),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(argmin(X0,X1,X2),arg=1)))).

assign(X0,X1,X2) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVar(X1,Y1)
             -> (is_BoolAssign(X2,Y2)
                 -> gecode_constraint_assign_30(Y0,Y1,Y2)
                 ;  throw(error(type_error('BoolAssign'(X2)),gecode_argument_error(assign(X0,X1,X2),arg=3))))
             ;  (is_FloatVar(X1,Y1)
                 -> (is_FloatAssign(X2,Y2)
                     -> gecode_constraint_assign_44(Y0,Y1,Y2)
                     ;  throw(error(type_error('FloatAssign'(X2)),gecode_argument_error(assign(X0,X1,X2),arg=3))))
                 ;  (is_IntVar(X1,Y1)
                     -> (is_IntAssign(X2,Y2)
                         -> gecode_constraint_assign_46(Y0,Y1,Y2)
                         ;  throw(error(type_error('IntAssign'(X2)),gecode_argument_error(assign(X0,X1,X2),arg=3))))
                     ;  (is_SetVar(X1,Y1)
                         -> (is_SetAssign(X2,Y2)
                             -> gecode_constraint_assign_48(Y0,Y1,Y2)
                             ;  throw(error(type_error('SetAssign'(X2)),gecode_argument_error(assign(X0,X1,X2),arg=3))))
                         ;  throw(error(type_error('SetVar'(X1)),gecode_argument_error(assign(X0,X1,X2),arg=2)))))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(assign(X0,X1,X2),arg=1)))).

assign(X0,X1,X2,X3) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVar(X1,Y1)
             -> (is_BoolAssign(X2,Y2)
                 -> (is_BoolVarValPrint(X3,Y3)
                     -> gecode_constraint_assign_31(Y0,Y1,Y2,Y3)
                     ;  throw(error(type_error('BoolVarValPrint'(X3)),gecode_argument_error(assign(X0,X1,X2,X3),arg=4))))
                 ;  throw(error(type_error('BoolAssign'(X2)),gecode_argument_error(assign(X0,X1,X2,X3),arg=3))))
             ;  (is_BoolVarArgs(X1,Y1)
                 -> (is_BoolVarBranch(X2,Y2)
                     -> (is_BoolAssign(X3,Y3)
                         -> gecode_constraint_assign_32(Y0,Y1,Y2,Y3)
                         ;  throw(error(type_error('BoolAssign'(X3)),gecode_argument_error(assign(X0,X1,X2,X3),arg=4))))
                     ;  throw(error(type_error('BoolVarBranch'(X2)),gecode_argument_error(assign(X0,X1,X2,X3),arg=3))))
                 ;  (is_FloatVarArgs(X1,Y1)
                     -> (is_FloatVarBranch(X2,Y2)
                         -> (is_FloatAssign(X3,Y3)
                             -> gecode_constraint_assign_35(Y0,Y1,Y2,Y3)
                             ;  throw(error(type_error('FloatAssign'(X3)),gecode_argument_error(assign(X0,X1,X2,X3),arg=4))))
                         ;  throw(error(type_error('FloatVarBranch'(X2)),gecode_argument_error(assign(X0,X1,X2,X3),arg=3))))
                     ;  (is_IntVarArgs(X1,Y1)
                         -> (is_IntVarBranch(X2,Y2)
                             -> (is_IntAssign(X3,Y3)
                                 -> gecode_constraint_assign_38(Y0,Y1,Y2,Y3)
                                 ;  throw(error(type_error('IntAssign'(X3)),gecode_argument_error(assign(X0,X1,X2,X3),arg=4))))
                             ;  throw(error(type_error('IntVarBranch'(X2)),gecode_argument_error(assign(X0,X1,X2,X3),arg=3))))
                         ;  (is_SetVarArgs(X1,Y1)
                             -> (is_SetVarBranch(X2,Y2)
                                 -> (is_SetAssign(X3,Y3)
                                     -> gecode_constraint_assign_41(Y0,Y1,Y2,Y3)
                                     ;  throw(error(type_error('SetAssign'(X3)),gecode_argument_error(assign(X0,X1,X2,X3),arg=4))))
                                 ;  throw(error(type_error('SetVarBranch'(X2)),gecode_argument_error(assign(X0,X1,X2,X3),arg=3))))
                             ;  (is_FloatVar(X1,Y1)
                                 -> (is_FloatAssign(X2,Y2)
                                     -> (is_FloatVarValPrint(X3,Y3)
                                         -> gecode_constraint_assign_45(Y0,Y1,Y2,Y3)
                                         ;  throw(error(type_error('FloatVarValPrint'(X3)),gecode_argument_error(assign(X0,X1,X2,X3),arg=4))))
                                     ;  throw(error(type_error('FloatAssign'(X2)),gecode_argument_error(assign(X0,X1,X2,X3),arg=3))))
                                 ;  (is_IntVar(X1,Y1)
                                     -> (is_IntAssign(X2,Y2)
                                         -> (is_IntVarValPrint(X3,Y3)
                                             -> gecode_constraint_assign_47(Y0,Y1,Y2,Y3)
                                             ;  throw(error(type_error('IntVarValPrint'(X3)),gecode_argument_error(assign(X0,X1,X2,X3),arg=4))))
                                         ;  throw(error(type_error('IntAssign'(X2)),gecode_argument_error(assign(X0,X1,X2,X3),arg=3))))
                                     ;  (is_SetVar(X1,Y1)
                                         -> (is_SetAssign(X2,Y2)
                                             -> (is_SetVarValPrint(X3,Y3)
                                                 -> gecode_constraint_assign_49(Y0,Y1,Y2,Y3)
                                                 ;  throw(error(type_error('SetVarValPrint'(X3)),gecode_argument_error(assign(X0,X1,X2,X3),arg=4))))
                                             ;  throw(error(type_error('SetAssign'(X2)),gecode_argument_error(assign(X0,X1,X2,X3),arg=3))))
                                         ;  throw(error(type_error('SetVar'(X1)),gecode_argument_error(assign(X0,X1,X2,X3),arg=2)))))))))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(assign(X0,X1,X2,X3),arg=1)))).

assign(X0,X1,X2,X3,X4) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVarArgs(X1,Y1)
             -> (is_BoolVarBranch(X2,Y2)
                 -> (is_BoolAssign(X3,Y3)
                     -> (is_BoolBranchFilter(X4,Y4)
                         -> gecode_constraint_assign_33(Y0,Y1,Y2,Y3,Y4)
                         ;  throw(error(type_error('BoolBranchFilter'(X4)),gecode_argument_error(assign(X0,X1,X2,X3,X4),arg=5))))
                     ;  throw(error(type_error('BoolAssign'(X3)),gecode_argument_error(assign(X0,X1,X2,X3,X4),arg=4))))
                 ;  throw(error(type_error('BoolVarBranch'(X2)),gecode_argument_error(assign(X0,X1,X2,X3,X4),arg=3))))
             ;  (is_FloatVarArgs(X1,Y1)
                 -> (is_FloatVarBranch(X2,Y2)
                     -> (is_FloatAssign(X3,Y3)
                         -> (is_FloatBranchFilter(X4,Y4)
                             -> gecode_constraint_assign_36(Y0,Y1,Y2,Y3,Y4)
                             ;  throw(error(type_error('FloatBranchFilter'(X4)),gecode_argument_error(assign(X0,X1,X2,X3,X4),arg=5))))
                         ;  throw(error(type_error('FloatAssign'(X3)),gecode_argument_error(assign(X0,X1,X2,X3,X4),arg=4))))
                     ;  throw(error(type_error('FloatVarBranch'(X2)),gecode_argument_error(assign(X0,X1,X2,X3,X4),arg=3))))
                 ;  (is_IntVarArgs(X1,Y1)
                     -> (is_IntVarBranch(X2,Y2)
                         -> (is_IntAssign(X3,Y3)
                             -> (is_IntBranchFilter(X4,Y4)
                                 -> gecode_constraint_assign_39(Y0,Y1,Y2,Y3,Y4)
                                 ;  throw(error(type_error('IntBranchFilter'(X4)),gecode_argument_error(assign(X0,X1,X2,X3,X4),arg=5))))
                             ;  throw(error(type_error('IntAssign'(X3)),gecode_argument_error(assign(X0,X1,X2,X3,X4),arg=4))))
                         ;  throw(error(type_error('IntVarBranch'(X2)),gecode_argument_error(assign(X0,X1,X2,X3,X4),arg=3))))
                     ;  (is_SetVarArgs(X1,Y1)
                         -> (is_SetVarBranch(X2,Y2)
                             -> (is_SetAssign(X3,Y3)
                                 -> (is_SetBranchFilter(X4,Y4)
                                     -> gecode_constraint_assign_42(Y0,Y1,Y2,Y3,Y4)
                                     ;  throw(error(type_error('SetBranchFilter'(X4)),gecode_argument_error(assign(X0,X1,X2,X3,X4),arg=5))))
                                 ;  throw(error(type_error('SetAssign'(X3)),gecode_argument_error(assign(X0,X1,X2,X3,X4),arg=4))))
                             ;  throw(error(type_error('SetVarBranch'(X2)),gecode_argument_error(assign(X0,X1,X2,X3,X4),arg=3))))
                         ;  throw(error(type_error('SetVarArgs'(X1)),gecode_argument_error(assign(X0,X1,X2,X3,X4),arg=2)))))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(assign(X0,X1,X2,X3,X4),arg=1)))).

assign(X0,X1,X2,X3,X4,X5) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVarArgs(X1,Y1)
             -> (is_BoolVarBranch(X2,Y2)
                 -> (is_BoolAssign(X3,Y3)
                     -> (is_BoolBranchFilter(X4,Y4)
                         -> (is_BoolVarValPrint(X5,Y5)
                             -> gecode_constraint_assign_34(Y0,Y1,Y2,Y3,Y4,Y5)
                             ;  throw(error(type_error('BoolVarValPrint'(X5)),gecode_argument_error(assign(X0,X1,X2,X3,X4,X5),arg=6))))
                         ;  throw(error(type_error('BoolBranchFilter'(X4)),gecode_argument_error(assign(X0,X1,X2,X3,X4,X5),arg=5))))
                     ;  throw(error(type_error('BoolAssign'(X3)),gecode_argument_error(assign(X0,X1,X2,X3,X4,X5),arg=4))))
                 ;  throw(error(type_error('BoolVarBranch'(X2)),gecode_argument_error(assign(X0,X1,X2,X3,X4,X5),arg=3))))
             ;  (is_FloatVarArgs(X1,Y1)
                 -> (is_FloatVarBranch(X2,Y2)
                     -> (is_FloatAssign(X3,Y3)
                         -> (is_FloatBranchFilter(X4,Y4)
                             -> (is_FloatVarValPrint(X5,Y5)
                                 -> gecode_constraint_assign_37(Y0,Y1,Y2,Y3,Y4,Y5)
                                 ;  throw(error(type_error('FloatVarValPrint'(X5)),gecode_argument_error(assign(X0,X1,X2,X3,X4,X5),arg=6))))
                             ;  throw(error(type_error('FloatBranchFilter'(X4)),gecode_argument_error(assign(X0,X1,X2,X3,X4,X5),arg=5))))
                         ;  throw(error(type_error('FloatAssign'(X3)),gecode_argument_error(assign(X0,X1,X2,X3,X4,X5),arg=4))))
                     ;  throw(error(type_error('FloatVarBranch'(X2)),gecode_argument_error(assign(X0,X1,X2,X3,X4,X5),arg=3))))
                 ;  (is_IntVarArgs(X1,Y1)
                     -> (is_IntVarBranch(X2,Y2)
                         -> (is_IntAssign(X3,Y3)
                             -> (is_IntBranchFilter(X4,Y4)
                                 -> (is_IntVarValPrint(X5,Y5)
                                     -> gecode_constraint_assign_40(Y0,Y1,Y2,Y3,Y4,Y5)
                                     ;  throw(error(type_error('IntVarValPrint'(X5)),gecode_argument_error(assign(X0,X1,X2,X3,X4,X5),arg=6))))
                                 ;  throw(error(type_error('IntBranchFilter'(X4)),gecode_argument_error(assign(X0,X1,X2,X3,X4,X5),arg=5))))
                             ;  throw(error(type_error('IntAssign'(X3)),gecode_argument_error(assign(X0,X1,X2,X3,X4,X5),arg=4))))
                         ;  throw(error(type_error('IntVarBranch'(X2)),gecode_argument_error(assign(X0,X1,X2,X3,X4,X5),arg=3))))
                     ;  (is_SetVarArgs(X1,Y1)
                         -> (is_SetVarBranch(X2,Y2)
                             -> (is_SetAssign(X3,Y3)
                                 -> (is_SetBranchFilter(X4,Y4)
                                     -> (is_SetVarValPrint(X5,Y5)
                                         -> gecode_constraint_assign_43(Y0,Y1,Y2,Y3,Y4,Y5)
                                         ;  throw(error(type_error('SetVarValPrint'(X5)),gecode_argument_error(assign(X0,X1,X2,X3,X4,X5),arg=6))))
                                     ;  throw(error(type_error('SetBranchFilter'(X4)),gecode_argument_error(assign(X0,X1,X2,X3,X4,X5),arg=5))))
                                 ;  throw(error(type_error('SetAssign'(X3)),gecode_argument_error(assign(X0,X1,X2,X3,X4,X5),arg=4))))
                             ;  throw(error(type_error('SetVarBranch'(X2)),gecode_argument_error(assign(X0,X1,X2,X3,X4,X5),arg=3))))
                         ;  throw(error(type_error('SetVarArgs'(X1)),gecode_argument_error(assign(X0,X1,X2,X3,X4,X5),arg=2)))))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(assign(X0,X1,X2,X3,X4,X5),arg=1)))).

binpacking(X0,X1,X2,X3) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_IntVarArgs(X1,Y1)
             -> (is_IntVarArgs(X2,Y2)
                 -> (is_IntArgs(X3,Y3)
                     -> gecode_constraint_binpacking_50(Y0,Y1,Y2,Y3)
                     ;  throw(error(type_error('IntArgs'(X3)),gecode_argument_error(binpacking(X0,X1,X2,X3),arg=4))))
                 ;  throw(error(type_error('IntVarArgs'(X2)),gecode_argument_error(binpacking(X0,X1,X2,X3),arg=3))))
             ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(binpacking(X0,X1,X2,X3),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(binpacking(X0,X1,X2,X3),arg=1)))).

branch(X0,X1,X2) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVar(X1,Y1)
             -> (is_BoolValBranch(X2,Y2)
                 -> gecode_constraint_branch_51(Y0,Y1,Y2)
                 ;  throw(error(type_error('BoolValBranch'(X2)),gecode_argument_error(branch(X0,X1,X2),arg=3))))
             ;  (is_FloatVar(X1,Y1)
                 -> (is_FloatValBranch(X2,Y2)
                     -> gecode_constraint_branch_74(Y0,Y1,Y2)
                     ;  throw(error(type_error('FloatValBranch'(X2)),gecode_argument_error(branch(X0,X1,X2),arg=3))))
                 ;  (is_IntVar(X1,Y1)
                     -> (is_IntValBranch(X2,Y2)
                         -> gecode_constraint_branch_76(Y0,Y1,Y2)
                         ;  throw(error(type_error('IntValBranch'(X2)),gecode_argument_error(branch(X0,X1,X2),arg=3))))
                     ;  (is_SetVar(X1,Y1)
                         -> (is_SetValBranch(X2,Y2)
                             -> gecode_constraint_branch_78(Y0,Y1,Y2)
                             ;  throw(error(type_error('SetValBranch'(X2)),gecode_argument_error(branch(X0,X1,X2),arg=3))))
                         ;  throw(error(type_error('SetVar'(X1)),gecode_argument_error(branch(X0,X1,X2),arg=2)))))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(branch(X0,X1,X2),arg=1)))).

branch(X0,X1,X2,X3) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVar(X1,Y1)
             -> (is_BoolValBranch(X2,Y2)
                 -> (is_BoolVarValPrint(X3,Y3)
                     -> gecode_constraint_branch_52(Y0,Y1,Y2,Y3)
                     ;  throw(error(type_error('BoolVarValPrint'(X3)),gecode_argument_error(branch(X0,X1,X2,X3),arg=4))))
                 ;  throw(error(type_error('BoolValBranch'(X2)),gecode_argument_error(branch(X0,X1,X2,X3),arg=3))))
             ;  (is_BoolVarArgs(X1,Y1)
                 -> (is_BoolVarBranch(X2,Y2)
                     -> (is_BoolValBranch(X3,Y3)
                         -> gecode_constraint_branch_53(Y0,Y1,Y2,Y3)
                         ;  throw(error(type_error('BoolValBranch'(X3)),gecode_argument_error(branch(X0,X1,X2,X3),arg=4))))
                     ;  throw(error(type_error('BoolVarBranch'(X2)),gecode_argument_error(branch(X0,X1,X2,X3),arg=3))))
                 ;  (is_FloatVarArgs(X1,Y1)
                     -> (is_FloatVarBranch(X2,Y2)
                         -> (is_FloatValBranch(X3,Y3)
                             -> gecode_constraint_branch_59(Y0,Y1,Y2,Y3)
                             ;  throw(error(type_error('FloatValBranch'(X3)),gecode_argument_error(branch(X0,X1,X2,X3),arg=4))))
                         ;  throw(error(type_error('FloatVarBranch'(X2)),gecode_argument_error(branch(X0,X1,X2,X3),arg=3))))
                     ;  (is_IntVarArgs(X1,Y1)
                         -> (is_IntVarBranch(X2,Y2)
                             -> (is_IntValBranch(X3,Y3)
                                 -> gecode_constraint_branch_65(Y0,Y1,Y2,Y3)
                                 ;  throw(error(type_error('IntValBranch'(X3)),gecode_argument_error(branch(X0,X1,X2,X3),arg=4))))
                             ;  throw(error(type_error('IntVarBranch'(X2)),gecode_argument_error(branch(X0,X1,X2,X3),arg=3))))
                         ;  (is_SetVarArgs(X1,Y1)
                             -> (is_SetVarBranch(X2,Y2)
                                 -> (is_SetValBranch(X3,Y3)
                                     -> gecode_constraint_branch_71(Y0,Y1,Y2,Y3)
                                     ;  throw(error(type_error('SetValBranch'(X3)),gecode_argument_error(branch(X0,X1,X2,X3),arg=4))))
                                 ;  throw(error(type_error('SetVarBranch'(X2)),gecode_argument_error(branch(X0,X1,X2,X3),arg=3))))
                             ;  (is_FloatVar(X1,Y1)
                                 -> (is_FloatValBranch(X2,Y2)
                                     -> (is_FloatVarValPrint(X3,Y3)
                                         -> gecode_constraint_branch_75(Y0,Y1,Y2,Y3)
                                         ;  throw(error(type_error('FloatVarValPrint'(X3)),gecode_argument_error(branch(X0,X1,X2,X3),arg=4))))
                                     ;  throw(error(type_error('FloatValBranch'(X2)),gecode_argument_error(branch(X0,X1,X2,X3),arg=3))))
                                 ;  (is_IntVar(X1,Y1)
                                     -> (is_IntValBranch(X2,Y2)
                                         -> (is_IntVarValPrint(X3,Y3)
                                             -> gecode_constraint_branch_77(Y0,Y1,Y2,Y3)
                                             ;  throw(error(type_error('IntVarValPrint'(X3)),gecode_argument_error(branch(X0,X1,X2,X3),arg=4))))
                                         ;  throw(error(type_error('IntValBranch'(X2)),gecode_argument_error(branch(X0,X1,X2,X3),arg=3))))
                                     ;  (is_SetVar(X1,Y1)
                                         -> (is_SetValBranch(X2,Y2)
                                             -> (is_SetVarValPrint(X3,Y3)
                                                 -> gecode_constraint_branch_79(Y0,Y1,Y2,Y3)
                                                 ;  throw(error(type_error('SetVarValPrint'(X3)),gecode_argument_error(branch(X0,X1,X2,X3),arg=4))))
                                             ;  throw(error(type_error('SetValBranch'(X2)),gecode_argument_error(branch(X0,X1,X2,X3),arg=3))))
                                         ;  throw(error(type_error('SetVar'(X1)),gecode_argument_error(branch(X0,X1,X2,X3),arg=2)))))))))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(branch(X0,X1,X2,X3),arg=1)))).

branch(X0,X1,X2,X3,X4) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVarArgs(X1,Y1)
             -> (is_BoolVarBranch(X2,Y2)
                 -> (is_BoolValBranch(X3,Y3)
                     -> (is_BoolBranchFilter(X4,Y4)
                         -> gecode_constraint_branch_54(Y0,Y1,Y2,Y3,Y4)
                         ;  (is_Symmetries(X4,Y4)
                             -> gecode_constraint_branch_56(Y0,Y1,Y2,Y3,Y4)
                             ;  throw(error(type_error('Symmetries'(X4)),gecode_argument_error(branch(X0,X1,X2,X3,X4),arg=5)))))
                     ;  throw(error(type_error('BoolValBranch'(X3)),gecode_argument_error(branch(X0,X1,X2,X3,X4),arg=4))))
                 ;  throw(error(type_error('BoolVarBranch'(X2)),gecode_argument_error(branch(X0,X1,X2,X3,X4),arg=3))))
             ;  (is_FloatVarArgs(X1,Y1)
                 -> (is_FloatVarBranch(X2,Y2)
                     -> (is_FloatValBranch(X3,Y3)
                         -> (is_FloatBranchFilter(X4,Y4)
                             -> gecode_constraint_branch_60(Y0,Y1,Y2,Y3,Y4)
                             ;  throw(error(type_error('FloatBranchFilter'(X4)),gecode_argument_error(branch(X0,X1,X2,X3,X4),arg=5))))
                         ;  throw(error(type_error('FloatValBranch'(X3)),gecode_argument_error(branch(X0,X1,X2,X3,X4),arg=4))))
                     ;  throw(error(type_error('FloatVarBranch'(X2)),gecode_argument_error(branch(X0,X1,X2,X3,X4),arg=3))))
                 ;  (is_IntVarArgs(X1,Y1)
                     -> (is_IntVarBranch(X2,Y2)
                         -> (is_IntValBranch(X3,Y3)
                             -> (is_Symmetries(X4,Y4)
                                 -> gecode_constraint_branch_62(Y0,Y1,Y2,Y3,Y4)
                                 ;  (is_IntBranchFilter(X4,Y4)
                                     -> gecode_constraint_branch_66(Y0,Y1,Y2,Y3,Y4)
                                     ;  throw(error(type_error('IntBranchFilter'(X4)),gecode_argument_error(branch(X0,X1,X2,X3,X4),arg=5)))))
                             ;  throw(error(type_error('IntValBranch'(X3)),gecode_argument_error(branch(X0,X1,X2,X3,X4),arg=4))))
                         ;  throw(error(type_error('IntVarBranch'(X2)),gecode_argument_error(branch(X0,X1,X2,X3,X4),arg=3))))
                     ;  (is_SetVarArgs(X1,Y1)
                         -> (is_SetVarBranch(X2,Y2)
                             -> (is_SetValBranch(X3,Y3)
                                 -> (is_Symmetries(X4,Y4)
                                     -> gecode_constraint_branch_68(Y0,Y1,Y2,Y3,Y4)
                                     ;  (is_SetBranchFilter(X4,Y4)
                                         -> gecode_constraint_branch_72(Y0,Y1,Y2,Y3,Y4)
                                         ;  throw(error(type_error('SetBranchFilter'(X4)),gecode_argument_error(branch(X0,X1,X2,X3,X4),arg=5)))))
                                 ;  throw(error(type_error('SetValBranch'(X3)),gecode_argument_error(branch(X0,X1,X2,X3,X4),arg=4))))
                             ;  throw(error(type_error('SetVarBranch'(X2)),gecode_argument_error(branch(X0,X1,X2,X3,X4),arg=3))))
                         ;  throw(error(type_error('SetVarArgs'(X1)),gecode_argument_error(branch(X0,X1,X2,X3,X4),arg=2)))))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(branch(X0,X1,X2,X3,X4),arg=1)))).

branch(X0,X1,X2,X3,X4,X5) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVarArgs(X1,Y1)
             -> (is_BoolVarBranch(X2,Y2)
                 -> (is_BoolValBranch(X3,Y3)
                     -> (is_BoolBranchFilter(X4,Y4)
                         -> (is_BoolVarValPrint(X5,Y5)
                             -> gecode_constraint_branch_55(Y0,Y1,Y2,Y3,Y4,Y5)
                             ;  throw(error(type_error('BoolVarValPrint'(X5)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5),arg=6))))
                         ;  (is_Symmetries(X4,Y4)
                             -> (is_BoolBranchFilter(X5,Y5)
                                 -> gecode_constraint_branch_57(Y0,Y1,Y2,Y3,Y4,Y5)
                                 ;  throw(error(type_error('BoolBranchFilter'(X5)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5),arg=6))))
                             ;  throw(error(type_error('Symmetries'(X4)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5),arg=5)))))
                     ;  throw(error(type_error('BoolValBranch'(X3)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5),arg=4))))
                 ;  throw(error(type_error('BoolVarBranch'(X2)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5),arg=3))))
             ;  (is_FloatVarArgs(X1,Y1)
                 -> (is_FloatVarBranch(X2,Y2)
                     -> (is_FloatValBranch(X3,Y3)
                         -> (is_FloatBranchFilter(X4,Y4)
                             -> (is_FloatVarValPrint(X5,Y5)
                                 -> gecode_constraint_branch_61(Y0,Y1,Y2,Y3,Y4,Y5)
                                 ;  throw(error(type_error('FloatVarValPrint'(X5)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5),arg=6))))
                             ;  throw(error(type_error('FloatBranchFilter'(X4)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5),arg=5))))
                         ;  throw(error(type_error('FloatValBranch'(X3)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5),arg=4))))
                     ;  throw(error(type_error('FloatVarBranch'(X2)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5),arg=3))))
                 ;  (is_IntVarArgs(X1,Y1)
                     -> (is_IntVarBranch(X2,Y2)
                         -> (is_IntValBranch(X3,Y3)
                             -> (is_Symmetries(X4,Y4)
                                 -> (is_IntBranchFilter(X5,Y5)
                                     -> gecode_constraint_branch_63(Y0,Y1,Y2,Y3,Y4,Y5)
                                     ;  throw(error(type_error('IntBranchFilter'(X5)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5),arg=6))))
                                 ;  (is_IntBranchFilter(X4,Y4)
                                     -> (is_IntVarValPrint(X5,Y5)
                                         -> gecode_constraint_branch_67(Y0,Y1,Y2,Y3,Y4,Y5)
                                         ;  throw(error(type_error('IntVarValPrint'(X5)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5),arg=6))))
                                     ;  throw(error(type_error('IntBranchFilter'(X4)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5),arg=5)))))
                             ;  throw(error(type_error('IntValBranch'(X3)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5),arg=4))))
                         ;  throw(error(type_error('IntVarBranch'(X2)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5),arg=3))))
                     ;  (is_SetVarArgs(X1,Y1)
                         -> (is_SetVarBranch(X2,Y2)
                             -> (is_SetValBranch(X3,Y3)
                                 -> (is_Symmetries(X4,Y4)
                                     -> (is_SetBranchFilter(X5,Y5)
                                         -> gecode_constraint_branch_69(Y0,Y1,Y2,Y3,Y4,Y5)
                                         ;  throw(error(type_error('SetBranchFilter'(X5)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5),arg=6))))
                                     ;  (is_SetBranchFilter(X4,Y4)
                                         -> (is_SetVarValPrint(X5,Y5)
                                             -> gecode_constraint_branch_73(Y0,Y1,Y2,Y3,Y4,Y5)
                                             ;  throw(error(type_error('SetVarValPrint'(X5)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5),arg=6))))
                                         ;  throw(error(type_error('SetBranchFilter'(X4)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5),arg=5)))))
                                 ;  throw(error(type_error('SetValBranch'(X3)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5),arg=4))))
                             ;  throw(error(type_error('SetVarBranch'(X2)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5),arg=3))))
                         ;  throw(error(type_error('SetVarArgs'(X1)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5),arg=2)))))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5),arg=1)))).

branch(X0,X1,X2,X3,X4,X5,X6) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVarArgs(X1,Y1)
             -> (is_BoolVarBranch(X2,Y2)
                 -> (is_BoolValBranch(X3,Y3)
                     -> (is_Symmetries(X4,Y4)
                         -> (is_BoolBranchFilter(X5,Y5)
                             -> (is_BoolVarValPrint(X6,Y6)
                                 -> gecode_constraint_branch_58(Y0,Y1,Y2,Y3,Y4,Y5,Y6)
                                 ;  throw(error(type_error('BoolVarValPrint'(X6)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5,X6),arg=7))))
                             ;  throw(error(type_error('BoolBranchFilter'(X5)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5,X6),arg=6))))
                         ;  throw(error(type_error('Symmetries'(X4)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5,X6),arg=5))))
                     ;  throw(error(type_error('BoolValBranch'(X3)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5,X6),arg=4))))
                 ;  throw(error(type_error('BoolVarBranch'(X2)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5,X6),arg=3))))
             ;  (is_IntVarArgs(X1,Y1)
                 -> (is_IntVarBranch(X2,Y2)
                     -> (is_IntValBranch(X3,Y3)
                         -> (is_Symmetries(X4,Y4)
                             -> (is_IntBranchFilter(X5,Y5)
                                 -> (is_IntVarValPrint(X6,Y6)
                                     -> gecode_constraint_branch_64(Y0,Y1,Y2,Y3,Y4,Y5,Y6)
                                     ;  throw(error(type_error('IntVarValPrint'(X6)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5,X6),arg=7))))
                                 ;  throw(error(type_error('IntBranchFilter'(X5)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5,X6),arg=6))))
                             ;  throw(error(type_error('Symmetries'(X4)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5,X6),arg=5))))
                         ;  throw(error(type_error('IntValBranch'(X3)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5,X6),arg=4))))
                     ;  throw(error(type_error('IntVarBranch'(X2)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5,X6),arg=3))))
                 ;  (is_SetVarArgs(X1,Y1)
                     -> (is_SetVarBranch(X2,Y2)
                         -> (is_SetValBranch(X3,Y3)
                             -> (is_Symmetries(X4,Y4)
                                 -> (is_SetBranchFilter(X5,Y5)
                                     -> (is_SetVarValPrint(X6,Y6)
                                         -> gecode_constraint_branch_70(Y0,Y1,Y2,Y3,Y4,Y5,Y6)
                                         ;  throw(error(type_error('SetVarValPrint'(X6)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5,X6),arg=7))))
                                     ;  throw(error(type_error('SetBranchFilter'(X5)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5,X6),arg=6))))
                                 ;  throw(error(type_error('Symmetries'(X4)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5,X6),arg=5))))
                             ;  throw(error(type_error('SetValBranch'(X3)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5,X6),arg=4))))
                         ;  throw(error(type_error('SetVarBranch'(X2)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5,X6),arg=3))))
                     ;  throw(error(type_error('SetVarArgs'(X1)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5,X6),arg=2))))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(branch(X0,X1,X2,X3,X4,X5,X6),arg=1)))).

branch(X0,X1) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_std_function(X1,Y1)
             -> gecode_constraint_branch_80(Y0,Y1)
             ;  throw(error(type_error('std::function<void(Space&home)>'(X1)),gecode_argument_error(branch(X0,X1),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(branch(X0,X1),arg=1)))).

cardinality(X0,X1,X2,X3) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_SetVarArgs(X1,Y1)
             -> (is_int(X2,Y2)
                 -> (is_int(X3,Y3)
                     -> gecode_constraint_cardinality_81(Y0,Y1,Y2,Y3)
                     ;  throw(error(type_error(int(X3)),gecode_argument_error(cardinality(X0,X1,X2,X3),arg=4))))
                 ;  throw(error(type_error(int(X2)),gecode_argument_error(cardinality(X0,X1,X2,X3),arg=3))))
             ;  (is_SetVar(X1,Y1)
                 -> (is_int(X2,Y2)
                     -> (is_int(X3,Y3)
                         -> gecode_constraint_cardinality_82(Y0,Y1,Y2,Y3)
                         ;  throw(error(type_error(int(X3)),gecode_argument_error(cardinality(X0,X1,X2,X3),arg=4))))
                     ;  throw(error(type_error(int(X2)),gecode_argument_error(cardinality(X0,X1,X2,X3),arg=3))))
                 ;  throw(error(type_error('SetVar'(X1)),gecode_argument_error(cardinality(X0,X1,X2,X3),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(cardinality(X0,X1,X2,X3),arg=1)))).

channel(X0,X1,X2) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVar(X1,Y1)
             -> (is_IntVar(X2,Y2)
                 -> gecode_constraint_channel_83(Y0,Y1,Y2)
                 ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(channel(X0,X1,X2),arg=3))))
             ;  (is_BoolVarArgs(X1,Y1)
                 -> (is_IntVar(X2,Y2)
                     -> gecode_constraint_channel_84(Y0,Y1,Y2)
                     ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(channel(X0,X1,X2),arg=3))))
                 ;  (is_IntVarArgs(X1,Y1)
                     -> (is_IntVarArgs(X2,Y2)
                         -> gecode_constraint_channel_86(Y0,Y1,Y2)
                         ;  throw(error(type_error('IntVarArgs'(X2)),gecode_argument_error(channel(X0,X1,X2),arg=3))))
                     ;  (is_FloatVar(X1,Y1)
                         -> (is_BoolVar(X2,Y2)
                             -> gecode_constraint_channel_88(Y0,Y1,Y2)
                             ;  (is_IntVar(X2,Y2)
                                 -> gecode_constraint_channel_89(Y0,Y1,Y2)
                                 ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(channel(X0,X1,X2),arg=3)))))
                         ;  throw(error(type_error('FloatVar'(X1)),gecode_argument_error(channel(X0,X1,X2),arg=2)))))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(channel(X0,X1,X2),arg=1)))).

channel(X0,X1,X2,X3) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVarArgs(X1,Y1)
             -> (is_IntVar(X2,Y2)
                 -> (is_int(X3,Y3)
                     -> gecode_constraint_channel_85(Y0,Y1,Y2,Y3)
                     ;  throw(error(type_error(int(X3)),gecode_argument_error(channel(X0,X1,X2,X3),arg=4))))
                 ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(channel(X0,X1,X2,X3),arg=3))))
             ;  throw(error(type_error('BoolVarArgs'(X1)),gecode_argument_error(channel(X0,X1,X2,X3),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(channel(X0,X1,X2,X3),arg=1)))).

channel(X0,X1,X2,X3,X4) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_IntVarArgs(X1,Y1)
             -> (is_int(X2,Y2)
                 -> (is_IntVarArgs(X3,Y3)
                     -> (is_int(X4,Y4)
                         -> gecode_constraint_channel_87(Y0,Y1,Y2,Y3,Y4)
                         ;  throw(error(type_error(int(X4)),gecode_argument_error(channel(X0,X1,X2,X3,X4),arg=5))))
                     ;  throw(error(type_error('IntVarArgs'(X3)),gecode_argument_error(channel(X0,X1,X2,X3,X4),arg=4))))
                 ;  throw(error(type_error(int(X2)),gecode_argument_error(channel(X0,X1,X2,X3,X4),arg=3))))
             ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(channel(X0,X1,X2,X3,X4),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(channel(X0,X1,X2,X3,X4),arg=1)))).

circuit(X0,X1,X2,X3,X4) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_IntArgs(X1,Y1)
             -> (is_IntVarArgs(X2,Y2)
                 -> (is_IntVarArgs(X3,Y3)
                     -> (is_IntVar(X4,Y4)
                         -> gecode_constraint_circuit_90(Y0,Y1,Y2,Y3,Y4)
                         ;  throw(error(type_error('IntVar'(X4)),gecode_argument_error(circuit(X0,X1,X2,X3,X4),arg=5))))
                     ;  throw(error(type_error('IntVarArgs'(X3)),gecode_argument_error(circuit(X0,X1,X2,X3,X4),arg=4))))
                 ;  (is_int(X2,Y2)
                     -> (is_IntVarArgs(X3,Y3)
                         -> (is_IntVar(X4,Y4)
                             -> gecode_constraint_circuit_93(Y0,Y1,Y2,Y3,Y4)
                             ;  throw(error(type_error('IntVar'(X4)),gecode_argument_error(circuit(X0,X1,X2,X3,X4),arg=5))))
                         ;  throw(error(type_error('IntVarArgs'(X3)),gecode_argument_error(circuit(X0,X1,X2,X3,X4),arg=4))))
                     ;  throw(error(type_error(int(X2)),gecode_argument_error(circuit(X0,X1,X2,X3,X4),arg=3)))))
             ;  throw(error(type_error('IntArgs'(X1)),gecode_argument_error(circuit(X0,X1,X2,X3,X4),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(circuit(X0,X1,X2,X3,X4),arg=1)))).

circuit(X0,X1,X2,X3) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_IntArgs(X1,Y1)
             -> (is_IntVarArgs(X2,Y2)
                 -> (is_IntVar(X3,Y3)
                     -> gecode_constraint_circuit_91(Y0,Y1,Y2,Y3)
                     ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(circuit(X0,X1,X2,X3),arg=4))))
                 ;  throw(error(type_error('IntVarArgs'(X2)),gecode_argument_error(circuit(X0,X1,X2,X3),arg=3))))
             ;  throw(error(type_error('IntArgs'(X1)),gecode_argument_error(circuit(X0,X1,X2,X3),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(circuit(X0,X1,X2,X3),arg=1)))).

circuit(X0,X1,X2,X3,X4,X5) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_IntArgs(X1,Y1)
             -> (is_int(X2,Y2)
                 -> (is_IntVarArgs(X3,Y3)
                     -> (is_IntVarArgs(X4,Y4)
                         -> (is_IntVar(X5,Y5)
                             -> gecode_constraint_circuit_92(Y0,Y1,Y2,Y3,Y4,Y5)
                             ;  throw(error(type_error('IntVar'(X5)),gecode_argument_error(circuit(X0,X1,X2,X3,X4,X5),arg=6))))
                         ;  throw(error(type_error('IntVarArgs'(X4)),gecode_argument_error(circuit(X0,X1,X2,X3,X4,X5),arg=5))))
                     ;  throw(error(type_error('IntVarArgs'(X3)),gecode_argument_error(circuit(X0,X1,X2,X3,X4,X5),arg=4))))
                 ;  throw(error(type_error(int(X2)),gecode_argument_error(circuit(X0,X1,X2,X3,X4,X5),arg=3))))
             ;  throw(error(type_error('IntArgs'(X1)),gecode_argument_error(circuit(X0,X1,X2,X3,X4,X5),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(circuit(X0,X1,X2,X3,X4,X5),arg=1)))).

circuit(X0,X1) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_IntVarArgs(X1,Y1)
             -> gecode_constraint_circuit_94(Y0,Y1)
             ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(circuit(X0,X1),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(circuit(X0,X1),arg=1)))).

circuit(X0,X1,X2) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_int(X1,Y1)
             -> (is_IntVarArgs(X2,Y2)
                 -> gecode_constraint_circuit_95(Y0,Y1,Y2)
                 ;  throw(error(type_error('IntVarArgs'(X2)),gecode_argument_error(circuit(X0,X1,X2),arg=3))))
             ;  throw(error(type_error(int(X1)),gecode_argument_error(circuit(X0,X1,X2),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(circuit(X0,X1,X2),arg=1)))).

clause(X0,X1,X2,X3,X4) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolOpType(X1,Y1)
             -> (is_BoolVarArgs(X2,Y2)
                 -> (is_BoolVarArgs(X3,Y3)
                     -> (is_BoolVar(X4,Y4)
                         -> gecode_constraint_clause_96(Y0,Y1,Y2,Y3,Y4)
                         ;  (is_int(X4,Y4)
                             -> gecode_constraint_clause_97(Y0,Y1,Y2,Y3,Y4)
                             ;  throw(error(type_error(int(X4)),gecode_argument_error(clause(X0,X1,X2,X3,X4),arg=5)))))
                     ;  throw(error(type_error('BoolVarArgs'(X3)),gecode_argument_error(clause(X0,X1,X2,X3,X4),arg=4))))
                 ;  throw(error(type_error('BoolVarArgs'(X2)),gecode_argument_error(clause(X0,X1,X2,X3,X4),arg=3))))
             ;  throw(error(type_error('BoolOpType'(X1)),gecode_argument_error(clause(X0,X1,X2,X3,X4),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(clause(X0,X1,X2,X3,X4),arg=1)))).

count(X0,X1,X2,X3,X4) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_IntVarArgs(X1,Y1)
             -> (is_IntArgs(X2,Y2)
                 -> (is_IntRelType(X3,Y3)
                     -> (is_int(X4,Y4)
                         -> gecode_constraint_count_98(Y0,Y1,Y2,Y3,Y4)
                         ;  (is_IntVar(X4,Y4)
                             -> gecode_constraint_count_99(Y0,Y1,Y2,Y3,Y4)
                             ;  throw(error(type_error('IntVar'(X4)),gecode_argument_error(count(X0,X1,X2,X3,X4),arg=5)))))
                     ;  throw(error(type_error('IntRelType'(X3)),gecode_argument_error(count(X0,X1,X2,X3,X4),arg=4))))
                 ;  (is_IntSet(X2,Y2)
                     -> (is_IntRelType(X3,Y3)
                         -> (is_int(X4,Y4)
                             -> gecode_constraint_count_103(Y0,Y1,Y2,Y3,Y4)
                             ;  (is_IntVar(X4,Y4)
                                 -> gecode_constraint_count_104(Y0,Y1,Y2,Y3,Y4)
                                 ;  throw(error(type_error('IntVar'(X4)),gecode_argument_error(count(X0,X1,X2,X3,X4),arg=5)))))
                         ;  throw(error(type_error('IntRelType'(X3)),gecode_argument_error(count(X0,X1,X2,X3,X4),arg=4))))
                     ;  (is_int(X2,Y2)
                         -> (is_IntRelType(X3,Y3)
                             -> (is_int(X4,Y4)
                                 -> gecode_constraint_count_107(Y0,Y1,Y2,Y3,Y4)
                                 ;  (is_IntVar(X4,Y4)
                                     -> gecode_constraint_count_108(Y0,Y1,Y2,Y3,Y4)
                                     ;  throw(error(type_error('IntVar'(X4)),gecode_argument_error(count(X0,X1,X2,X3,X4),arg=5)))))
                             ;  throw(error(type_error('IntRelType'(X3)),gecode_argument_error(count(X0,X1,X2,X3,X4),arg=4))))
                         ;  (is_IntVar(X2,Y2)
                             -> (is_IntRelType(X3,Y3)
                                 -> (is_int(X4,Y4)
                                     -> gecode_constraint_count_109(Y0,Y1,Y2,Y3,Y4)
                                     ;  (is_IntVar(X4,Y4)
                                         -> gecode_constraint_count_110(Y0,Y1,Y2,Y3,Y4)
                                         ;  throw(error(type_error('IntVar'(X4)),gecode_argument_error(count(X0,X1,X2,X3,X4),arg=5)))))
                                 ;  throw(error(type_error('IntRelType'(X3)),gecode_argument_error(count(X0,X1,X2,X3,X4),arg=4))))
                             ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(count(X0,X1,X2,X3,X4),arg=3)))))))
             ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(count(X0,X1,X2,X3,X4),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(count(X0,X1,X2,X3,X4),arg=1)))).

count(X0,X1,X2,X3) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_IntVarArgs(X1,Y1)
             -> (is_IntSetArgs(X2,Y2)
                 -> (is_IntArgs(X3,Y3)
                     -> gecode_constraint_count_100(Y0,Y1,Y2,Y3)
                     ;  throw(error(type_error('IntArgs'(X3)),gecode_argument_error(count(X0,X1,X2,X3),arg=4))))
                 ;  (is_IntSet(X2,Y2)
                     -> (is_IntArgs(X3,Y3)
                         -> gecode_constraint_count_102(Y0,Y1,Y2,Y3)
                         ;  throw(error(type_error('IntArgs'(X3)),gecode_argument_error(count(X0,X1,X2,X3),arg=4))))
                     ;  (is_IntVarArgs(X2,Y2)
                         -> (is_IntArgs(X3,Y3)
                             -> gecode_constraint_count_105(Y0,Y1,Y2,Y3)
                             ;  throw(error(type_error('IntArgs'(X3)),gecode_argument_error(count(X0,X1,X2,X3),arg=4))))
                         ;  throw(error(type_error('IntVarArgs'(X2)),gecode_argument_error(count(X0,X1,X2,X3),arg=3))))))
             ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(count(X0,X1,X2,X3),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(count(X0,X1,X2,X3),arg=1)))).

count(X0,X1,X2) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_IntVarArgs(X1,Y1)
             -> (is_IntSetArgs(X2,Y2)
                 -> gecode_constraint_count_101(Y0,Y1,Y2)
                 ;  (is_IntVarArgs(X2,Y2)
                     -> gecode_constraint_count_106(Y0,Y1,Y2)
                     ;  throw(error(type_error('IntVarArgs'(X2)),gecode_argument_error(count(X0,X1,X2),arg=3)))))
             ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(count(X0,X1,X2),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(count(X0,X1,X2),arg=1)))).

cumulative(X0,X1,X2,X3,X4,X5) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_int(X1,Y1)
             -> (is_IntVarArgs(X2,Y2)
                 -> (is_IntArgs(X3,Y3)
                     -> (is_IntArgs(X4,Y4)
                         -> (is_BoolVarArgs(X5,Y5)
                             -> gecode_constraint_cumulative_111(Y0,Y1,Y2,Y3,Y4,Y5)
                             ;  throw(error(type_error('BoolVarArgs'(X5)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5),arg=6))))
                         ;  throw(error(type_error('IntArgs'(X4)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5),arg=5))))
                     ;  (is_IntVarArgs(X3,Y3)
                         -> (is_IntVarArgs(X4,Y4)
                             -> (is_IntArgs(X5,Y5)
                                 -> gecode_constraint_cumulative_114(Y0,Y1,Y2,Y3,Y4,Y5)
                                 ;  throw(error(type_error('IntArgs'(X5)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5),arg=6))))
                             ;  throw(error(type_error('IntVarArgs'(X4)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5),arg=5))))
                         ;  throw(error(type_error('IntVarArgs'(X3)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5),arg=4)))))
                 ;  (is_TaskTypeArgs(X2,Y2)
                     -> (is_IntVarArgs(X3,Y3)
                         -> (is_IntArgs(X4,Y4)
                             -> (is_IntArgs(X5,Y5)
                                 -> gecode_constraint_cumulative_116(Y0,Y1,Y2,Y3,Y4,Y5)
                                 ;  throw(error(type_error('IntArgs'(X5)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5),arg=6))))
                             ;  throw(error(type_error('IntArgs'(X4)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5),arg=5))))
                         ;  throw(error(type_error('IntVarArgs'(X3)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5),arg=4))))
                     ;  throw(error(type_error('TaskTypeArgs'(X2)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5),arg=3)))))
             ;  (is_IntVar(X1,Y1)
                 -> (is_IntVarArgs(X2,Y2)
                     -> (is_IntArgs(X3,Y3)
                         -> (is_IntArgs(X4,Y4)
                             -> (is_BoolVarArgs(X5,Y5)
                                 -> gecode_constraint_cumulative_117(Y0,Y1,Y2,Y3,Y4,Y5)
                                 ;  throw(error(type_error('BoolVarArgs'(X5)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5),arg=6))))
                             ;  throw(error(type_error('IntArgs'(X4)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5),arg=5))))
                         ;  (is_IntVarArgs(X3,Y3)
                             -> (is_IntVarArgs(X4,Y4)
                                 -> (is_IntArgs(X5,Y5)
                                     -> gecode_constraint_cumulative_120(Y0,Y1,Y2,Y3,Y4,Y5)
                                     ;  throw(error(type_error('IntArgs'(X5)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5),arg=6))))
                                 ;  throw(error(type_error('IntVarArgs'(X4)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5),arg=5))))
                             ;  throw(error(type_error('IntVarArgs'(X3)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5),arg=4)))))
                     ;  (is_TaskTypeArgs(X2,Y2)
                         -> (is_IntVarArgs(X3,Y3)
                             -> (is_IntArgs(X4,Y4)
                                 -> (is_IntArgs(X5,Y5)
                                     -> gecode_constraint_cumulative_122(Y0,Y1,Y2,Y3,Y4,Y5)
                                     ;  throw(error(type_error('IntArgs'(X5)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5),arg=6))))
                                 ;  throw(error(type_error('IntArgs'(X4)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5),arg=5))))
                             ;  throw(error(type_error('IntVarArgs'(X3)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5),arg=4))))
                         ;  throw(error(type_error('TaskTypeArgs'(X2)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5),arg=3)))))
                 ;  throw(error(type_error('IntVar'(X1)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5),arg=1)))).

cumulative(X0,X1,X2,X3,X4) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_int(X1,Y1)
             -> (is_IntVarArgs(X2,Y2)
                 -> (is_IntArgs(X3,Y3)
                     -> (is_IntArgs(X4,Y4)
                         -> gecode_constraint_cumulative_112(Y0,Y1,Y2,Y3,Y4)
                         ;  throw(error(type_error('IntArgs'(X4)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4),arg=5))))
                     ;  throw(error(type_error('IntArgs'(X3)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4),arg=4))))
                 ;  throw(error(type_error('IntVarArgs'(X2)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4),arg=3))))
             ;  (is_IntVar(X1,Y1)
                 -> (is_IntVarArgs(X2,Y2)
                     -> (is_IntArgs(X3,Y3)
                         -> (is_IntArgs(X4,Y4)
                             -> gecode_constraint_cumulative_118(Y0,Y1,Y2,Y3,Y4)
                             ;  throw(error(type_error('IntArgs'(X4)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4),arg=5))))
                         ;  throw(error(type_error('IntArgs'(X3)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4),arg=4))))
                     ;  throw(error(type_error('IntVarArgs'(X2)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4),arg=3))))
                 ;  throw(error(type_error('IntVar'(X1)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4),arg=1)))).

cumulative(X0,X1,X2,X3,X4,X5,X6) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_int(X1,Y1)
             -> (is_IntVarArgs(X2,Y2)
                 -> (is_IntVarArgs(X3,Y3)
                     -> (is_IntVarArgs(X4,Y4)
                         -> (is_IntArgs(X5,Y5)
                             -> (is_BoolVarArgs(X6,Y6)
                                 -> gecode_constraint_cumulative_113(Y0,Y1,Y2,Y3,Y4,Y5,Y6)
                                 ;  throw(error(type_error('BoolVarArgs'(X6)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5,X6),arg=7))))
                             ;  throw(error(type_error('IntArgs'(X5)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5,X6),arg=6))))
                         ;  throw(error(type_error('IntVarArgs'(X4)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5,X6),arg=5))))
                     ;  throw(error(type_error('IntVarArgs'(X3)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5,X6),arg=4))))
                 ;  (is_TaskTypeArgs(X2,Y2)
                     -> (is_IntVarArgs(X3,Y3)
                         -> (is_IntArgs(X4,Y4)
                             -> (is_IntArgs(X5,Y5)
                                 -> (is_BoolVarArgs(X6,Y6)
                                     -> gecode_constraint_cumulative_115(Y0,Y1,Y2,Y3,Y4,Y5,Y6)
                                     ;  throw(error(type_error('BoolVarArgs'(X6)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5,X6),arg=7))))
                                 ;  throw(error(type_error('IntArgs'(X5)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5,X6),arg=6))))
                             ;  throw(error(type_error('IntArgs'(X4)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5,X6),arg=5))))
                         ;  throw(error(type_error('IntVarArgs'(X3)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5,X6),arg=4))))
                     ;  throw(error(type_error('TaskTypeArgs'(X2)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5,X6),arg=3)))))
             ;  (is_IntVar(X1,Y1)
                 -> (is_IntVarArgs(X2,Y2)
                     -> (is_IntVarArgs(X3,Y3)
                         -> (is_IntVarArgs(X4,Y4)
                             -> (is_IntArgs(X5,Y5)
                                 -> (is_BoolVarArgs(X6,Y6)
                                     -> gecode_constraint_cumulative_119(Y0,Y1,Y2,Y3,Y4,Y5,Y6)
                                     ;  throw(error(type_error('BoolVarArgs'(X6)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5,X6),arg=7))))
                                 ;  throw(error(type_error('IntArgs'(X5)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5,X6),arg=6))))
                             ;  throw(error(type_error('IntVarArgs'(X4)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5,X6),arg=5))))
                         ;  throw(error(type_error('IntVarArgs'(X3)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5,X6),arg=4))))
                     ;  (is_TaskTypeArgs(X2,Y2)
                         -> (is_IntVarArgs(X3,Y3)
                             -> (is_IntArgs(X4,Y4)
                                 -> (is_IntArgs(X5,Y5)
                                     -> (is_BoolVarArgs(X6,Y6)
                                         -> gecode_constraint_cumulative_121(Y0,Y1,Y2,Y3,Y4,Y5,Y6)
                                         ;  throw(error(type_error('BoolVarArgs'(X6)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5,X6),arg=7))))
                                     ;  throw(error(type_error('IntArgs'(X5)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5,X6),arg=6))))
                                 ;  throw(error(type_error('IntArgs'(X4)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5,X6),arg=5))))
                             ;  throw(error(type_error('IntVarArgs'(X3)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5,X6),arg=4))))
                         ;  throw(error(type_error('TaskTypeArgs'(X2)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5,X6),arg=3)))))
                 ;  throw(error(type_error('IntVar'(X1)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5,X6),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(cumulative(X0,X1,X2,X3,X4,X5,X6),arg=1)))).

cumulatives(X0,X1,X2,X3,X4,X5,X6,X7) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_IntArgs(X1,Y1)
             -> (is_IntVarArgs(X2,Y2)
                 -> (is_IntArgs(X3,Y3)
                     -> (is_IntVarArgs(X4,Y4)
                         -> (is_IntArgs(X5,Y5)
                             -> (is_IntArgs(X6,Y6)
                                 -> (is_bool(X7,Y7)
                                     -> gecode_constraint_cumulatives_123(Y0,Y1,Y2,Y3,Y4,Y5,Y6,Y7)
                                     ;  throw(error(type_error(bool(X7)),gecode_argument_error(cumulatives(X0,X1,X2,X3,X4,X5,X6,X7),arg=8))))
                                 ;  throw(error(type_error('IntArgs'(X6)),gecode_argument_error(cumulatives(X0,X1,X2,X3,X4,X5,X6,X7),arg=7))))
                             ;  (is_IntVarArgs(X5,Y5)
                                 -> (is_IntArgs(X6,Y6)
                                     -> (is_bool(X7,Y7)
                                         -> gecode_constraint_cumulatives_124(Y0,Y1,Y2,Y3,Y4,Y5,Y6,Y7)
                                         ;  throw(error(type_error(bool(X7)),gecode_argument_error(cumulatives(X0,X1,X2,X3,X4,X5,X6,X7),arg=8))))
                                     ;  throw(error(type_error('IntArgs'(X6)),gecode_argument_error(cumulatives(X0,X1,X2,X3,X4,X5,X6,X7),arg=7))))
                                 ;  throw(error(type_error('IntVarArgs'(X5)),gecode_argument_error(cumulatives(X0,X1,X2,X3,X4,X5,X6,X7),arg=6)))))
                         ;  throw(error(type_error('IntVarArgs'(X4)),gecode_argument_error(cumulatives(X0,X1,X2,X3,X4,X5,X6,X7),arg=5))))
                     ;  (is_IntVarArgs(X3,Y3)
                         -> (is_IntVarArgs(X4,Y4)
                             -> (is_IntArgs(X5,Y5)
                                 -> (is_IntArgs(X6,Y6)
                                     -> (is_bool(X7,Y7)
                                         -> gecode_constraint_cumulatives_125(Y0,Y1,Y2,Y3,Y4,Y5,Y6,Y7)
                                         ;  throw(error(type_error(bool(X7)),gecode_argument_error(cumulatives(X0,X1,X2,X3,X4,X5,X6,X7),arg=8))))
                                     ;  throw(error(type_error('IntArgs'(X6)),gecode_argument_error(cumulatives(X0,X1,X2,X3,X4,X5,X6,X7),arg=7))))
                                 ;  (is_IntVarArgs(X5,Y5)
                                     -> (is_IntArgs(X6,Y6)
                                         -> (is_bool(X7,Y7)
                                             -> gecode_constraint_cumulatives_126(Y0,Y1,Y2,Y3,Y4,Y5,Y6,Y7)
                                             ;  throw(error(type_error(bool(X7)),gecode_argument_error(cumulatives(X0,X1,X2,X3,X4,X5,X6,X7),arg=8))))
                                         ;  throw(error(type_error('IntArgs'(X6)),gecode_argument_error(cumulatives(X0,X1,X2,X3,X4,X5,X6,X7),arg=7))))
                                     ;  throw(error(type_error('IntVarArgs'(X5)),gecode_argument_error(cumulatives(X0,X1,X2,X3,X4,X5,X6,X7),arg=6)))))
                             ;  throw(error(type_error('IntVarArgs'(X4)),gecode_argument_error(cumulatives(X0,X1,X2,X3,X4,X5,X6,X7),arg=5))))
                         ;  throw(error(type_error('IntVarArgs'(X3)),gecode_argument_error(cumulatives(X0,X1,X2,X3,X4,X5,X6,X7),arg=4)))))
                 ;  throw(error(type_error('IntVarArgs'(X2)),gecode_argument_error(cumulatives(X0,X1,X2,X3,X4,X5,X6,X7),arg=3))))
             ;  (is_IntVarArgs(X1,Y1)
                 -> (is_IntVarArgs(X2,Y2)
                     -> (is_IntArgs(X3,Y3)
                         -> (is_IntVarArgs(X4,Y4)
                             -> (is_IntArgs(X5,Y5)
                                 -> (is_IntArgs(X6,Y6)
                                     -> (is_bool(X7,Y7)
                                         -> gecode_constraint_cumulatives_127(Y0,Y1,Y2,Y3,Y4,Y5,Y6,Y7)
                                         ;  throw(error(type_error(bool(X7)),gecode_argument_error(cumulatives(X0,X1,X2,X3,X4,X5,X6,X7),arg=8))))
                                     ;  throw(error(type_error('IntArgs'(X6)),gecode_argument_error(cumulatives(X0,X1,X2,X3,X4,X5,X6,X7),arg=7))))
                                 ;  (is_IntVarArgs(X5,Y5)
                                     -> (is_IntArgs(X6,Y6)
                                         -> (is_bool(X7,Y7)
                                             -> gecode_constraint_cumulatives_128(Y0,Y1,Y2,Y3,Y4,Y5,Y6,Y7)
                                             ;  throw(error(type_error(bool(X7)),gecode_argument_error(cumulatives(X0,X1,X2,X3,X4,X5,X6,X7),arg=8))))
                                         ;  throw(error(type_error('IntArgs'(X6)),gecode_argument_error(cumulatives(X0,X1,X2,X3,X4,X5,X6,X7),arg=7))))
                                     ;  throw(error(type_error('IntVarArgs'(X5)),gecode_argument_error(cumulatives(X0,X1,X2,X3,X4,X5,X6,X7),arg=6)))))
                             ;  throw(error(type_error('IntVarArgs'(X4)),gecode_argument_error(cumulatives(X0,X1,X2,X3,X4,X5,X6,X7),arg=5))))
                         ;  (is_IntVarArgs(X3,Y3)
                             -> (is_IntVarArgs(X4,Y4)
                                 -> (is_IntArgs(X5,Y5)
                                     -> (is_IntArgs(X6,Y6)
                                         -> (is_bool(X7,Y7)
                                             -> gecode_constraint_cumulatives_129(Y0,Y1,Y2,Y3,Y4,Y5,Y6,Y7)
                                             ;  throw(error(type_error(bool(X7)),gecode_argument_error(cumulatives(X0,X1,X2,X3,X4,X5,X6,X7),arg=8))))
                                         ;  throw(error(type_error('IntArgs'(X6)),gecode_argument_error(cumulatives(X0,X1,X2,X3,X4,X5,X6,X7),arg=7))))
                                     ;  (is_IntVarArgs(X5,Y5)
                                         -> (is_IntArgs(X6,Y6)
                                             -> (is_bool(X7,Y7)
                                                 -> gecode_constraint_cumulatives_130(Y0,Y1,Y2,Y3,Y4,Y5,Y6,Y7)
                                                 ;  throw(error(type_error(bool(X7)),gecode_argument_error(cumulatives(X0,X1,X2,X3,X4,X5,X6,X7),arg=8))))
                                             ;  throw(error(type_error('IntArgs'(X6)),gecode_argument_error(cumulatives(X0,X1,X2,X3,X4,X5,X6,X7),arg=7))))
                                         ;  throw(error(type_error('IntVarArgs'(X5)),gecode_argument_error(cumulatives(X0,X1,X2,X3,X4,X5,X6,X7),arg=6)))))
                                 ;  throw(error(type_error('IntVarArgs'(X4)),gecode_argument_error(cumulatives(X0,X1,X2,X3,X4,X5,X6,X7),arg=5))))
                             ;  throw(error(type_error('IntVarArgs'(X3)),gecode_argument_error(cumulatives(X0,X1,X2,X3,X4,X5,X6,X7),arg=4)))))
                     ;  throw(error(type_error('IntVarArgs'(X2)),gecode_argument_error(cumulatives(X0,X1,X2,X3,X4,X5,X6,X7),arg=3))))
                 ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(cumulatives(X0,X1,X2,X3,X4,X5,X6,X7),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(cumulatives(X0,X1,X2,X3,X4,X5,X6,X7),arg=1)))).

distinct(X0,X1,X2) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVarArgs(X1,Y1)
             -> (is_IntVarArgs(X2,Y2)
                 -> gecode_constraint_distinct_131(Y0,Y1,Y2)
                 ;  throw(error(type_error('IntVarArgs'(X2)),gecode_argument_error(distinct(X0,X1,X2),arg=3))))
             ;  (is_IntArgs(X1,Y1)
                 -> (is_IntVarArgs(X2,Y2)
                     -> gecode_constraint_distinct_132(Y0,Y1,Y2)
                     ;  throw(error(type_error('IntVarArgs'(X2)),gecode_argument_error(distinct(X0,X1,X2),arg=3))))
                 ;  (is_IntVarArgs(X1,Y1)
                     -> (is_int(X2,Y2)
                         -> gecode_constraint_distinct_133(Y0,Y1,Y2)
                         ;  throw(error(type_error(int(X2)),gecode_argument_error(distinct(X0,X1,X2),arg=3))))
                     ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(distinct(X0,X1,X2),arg=2))))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(distinct(X0,X1,X2),arg=1)))).

distinct(X0,X1) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_IntVarArgs(X1,Y1)
             -> gecode_constraint_distinct_134(Y0,Y1)
             ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(distinct(X0,X1),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(distinct(X0,X1),arg=1)))).

div(X0,X1,X2,X3) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_FloatVar(X1,Y1)
             -> (is_FloatVar(X2,Y2)
                 -> (is_FloatVar(X3,Y3)
                     -> gecode_constraint_div_135(Y0,Y1,Y2,Y3)
                     ;  throw(error(type_error('FloatVar'(X3)),gecode_argument_error(div(X0,X1,X2,X3),arg=4))))
                 ;  throw(error(type_error('FloatVar'(X2)),gecode_argument_error(div(X0,X1,X2,X3),arg=3))))
             ;  (is_IntVar(X1,Y1)
                 -> (is_IntVar(X2,Y2)
                     -> (is_IntVar(X3,Y3)
                         -> gecode_constraint_div_136(Y0,Y1,Y2,Y3)
                         ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(div(X0,X1,X2,X3),arg=4))))
                     ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(div(X0,X1,X2,X3),arg=3))))
                 ;  throw(error(type_error('IntVar'(X1)),gecode_argument_error(div(X0,X1,X2,X3),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(div(X0,X1,X2,X3),arg=1)))).

divmod(X0,X1,X2,X3,X4) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_IntVar(X1,Y1)
             -> (is_IntVar(X2,Y2)
                 -> (is_IntVar(X3,Y3)
                     -> (is_IntVar(X4,Y4)
                         -> gecode_constraint_divmod_137(Y0,Y1,Y2,Y3,Y4)
                         ;  throw(error(type_error('IntVar'(X4)),gecode_argument_error(divmod(X0,X1,X2,X3,X4),arg=5))))
                     ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(divmod(X0,X1,X2,X3,X4),arg=4))))
                 ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(divmod(X0,X1,X2,X3,X4),arg=3))))
             ;  throw(error(type_error('IntVar'(X1)),gecode_argument_error(divmod(X0,X1,X2,X3,X4),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(divmod(X0,X1,X2,X3,X4),arg=1)))).

dom(X0,X1,X2,X3) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_FloatVarArgs(X1,Y1)
             -> (is_FloatNum(X2,Y2)
                 -> (is_FloatNum(X3,Y3)
                     -> gecode_constraint_dom_141(Y0,Y1,Y2,Y3)
                     ;  throw(error(type_error('FloatNum'(X3)),gecode_argument_error(dom(X0,X1,X2,X3),arg=4))))
                 ;  throw(error(type_error('FloatNum'(X2)),gecode_argument_error(dom(X0,X1,X2,X3),arg=3))))
             ;  (is_IntVarArgs(X1,Y1)
                 -> (is_int(X2,Y2)
                     -> (is_int(X3,Y3)
                         -> gecode_constraint_dom_145(Y0,Y1,Y2,Y3)
                         ;  throw(error(type_error(int(X3)),gecode_argument_error(dom(X0,X1,X2,X3),arg=4))))
                     ;  throw(error(type_error(int(X2)),gecode_argument_error(dom(X0,X1,X2,X3),arg=3))))
                 ;  (is_SetVarArgs(X1,Y1)
                     -> (is_SetRelType(X2,Y2)
                         -> (is_IntSet(X3,Y3)
                             -> gecode_constraint_dom_148(Y0,Y1,Y2,Y3)
                             ;  (is_int(X3,Y3)
                                 -> gecode_constraint_dom_149(Y0,Y1,Y2,Y3)
                                 ;  throw(error(type_error(int(X3)),gecode_argument_error(dom(X0,X1,X2,X3),arg=4)))))
                         ;  throw(error(type_error('SetRelType'(X2)),gecode_argument_error(dom(X0,X1,X2,X3),arg=3))))
                     ;  (is_FloatVar(X1,Y1)
                         -> (is_FloatNum(X2,Y2)
                             -> (is_FloatNum(X3,Y3)
                                 -> gecode_constraint_dom_151(Y0,Y1,Y2,Y3)
                                 ;  throw(error(type_error('FloatNum'(X3)),gecode_argument_error(dom(X0,X1,X2,X3),arg=4))))
                             ;  (is_FloatVal(X2,Y2)
                                 -> (is_Reify(X3,Y3)
                                     -> gecode_constraint_dom_154(Y0,Y1,Y2,Y3)
                                     ;  throw(error(type_error('Reify'(X3)),gecode_argument_error(dom(X0,X1,X2,X3),arg=4))))
                                 ;  throw(error(type_error('FloatVal'(X2)),gecode_argument_error(dom(X0,X1,X2,X3),arg=3)))))
                         ;  (is_IntVar(X1,Y1)
                             -> (is_IntSet(X2,Y2)
                                 -> (is_Reify(X3,Y3)
                                     -> gecode_constraint_dom_157(Y0,Y1,Y2,Y3)
                                     ;  throw(error(type_error('Reify'(X3)),gecode_argument_error(dom(X0,X1,X2,X3),arg=4))))
                                 ;  (is_int(X2,Y2)
                                     -> (is_int(X3,Y3)
                                         -> gecode_constraint_dom_158(Y0,Y1,Y2,Y3)
                                         ;  (is_Reify(X3,Y3)
                                             -> gecode_constraint_dom_161(Y0,Y1,Y2,Y3)
                                             ;  throw(error(type_error('Reify'(X3)),gecode_argument_error(dom(X0,X1,X2,X3),arg=4)))))
                                     ;  throw(error(type_error(int(X2)),gecode_argument_error(dom(X0,X1,X2,X3),arg=3)))))
                             ;  (is_SetVar(X1,Y1)
                                 -> (is_SetRelType(X2,Y2)
                                     -> (is_IntSet(X3,Y3)
                                         -> gecode_constraint_dom_163(Y0,Y1,Y2,Y3)
                                         ;  (is_int(X3,Y3)
                                             -> gecode_constraint_dom_165(Y0,Y1,Y2,Y3)
                                             ;  throw(error(type_error(int(X3)),gecode_argument_error(dom(X0,X1,X2,X3),arg=4)))))
                                     ;  throw(error(type_error('SetRelType'(X2)),gecode_argument_error(dom(X0,X1,X2,X3),arg=3))))
                                 ;  throw(error(type_error('SetVar'(X1)),gecode_argument_error(dom(X0,X1,X2,X3),arg=2)))))))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(dom(X0,X1,X2,X3),arg=1)))).

dom(X0,X1,X2,X3,X4) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_SetVarArgs(X1,Y1)
             -> (is_SetRelType(X2,Y2)
                 -> (is_int(X3,Y3)
                     -> (is_int(X4,Y4)
                         -> gecode_constraint_dom_150(Y0,Y1,Y2,Y3,Y4)
                         ;  throw(error(type_error(int(X4)),gecode_argument_error(dom(X0,X1,X2,X3,X4),arg=5))))
                     ;  throw(error(type_error(int(X3)),gecode_argument_error(dom(X0,X1,X2,X3,X4),arg=4))))
                 ;  throw(error(type_error('SetRelType'(X2)),gecode_argument_error(dom(X0,X1,X2,X3,X4),arg=3))))
             ;  (is_FloatVar(X1,Y1)
                 -> (is_FloatNum(X2,Y2)
                     -> (is_FloatNum(X3,Y3)
                         -> (is_Reify(X4,Y4)
                             -> gecode_constraint_dom_152(Y0,Y1,Y2,Y3,Y4)
                             ;  throw(error(type_error('Reify'(X4)),gecode_argument_error(dom(X0,X1,X2,X3,X4),arg=5))))
                         ;  throw(error(type_error('FloatNum'(X3)),gecode_argument_error(dom(X0,X1,X2,X3,X4),arg=4))))
                     ;  throw(error(type_error('FloatNum'(X2)),gecode_argument_error(dom(X0,X1,X2,X3,X4),arg=3))))
                 ;  (is_IntVar(X1,Y1)
                     -> (is_int(X2,Y2)
                         -> (is_int(X3,Y3)
                             -> (is_Reify(X4,Y4)
                                 -> gecode_constraint_dom_160(Y0,Y1,Y2,Y3,Y4)
                                 ;  throw(error(type_error('Reify'(X4)),gecode_argument_error(dom(X0,X1,X2,X3,X4),arg=5))))
                             ;  throw(error(type_error(int(X3)),gecode_argument_error(dom(X0,X1,X2,X3,X4),arg=4))))
                         ;  throw(error(type_error(int(X2)),gecode_argument_error(dom(X0,X1,X2,X3,X4),arg=3))))
                     ;  (is_SetVar(X1,Y1)
                         -> (is_SetRelType(X2,Y2)
                             -> (is_IntSet(X3,Y3)
                                 -> (is_Reify(X4,Y4)
                                     -> gecode_constraint_dom_164(Y0,Y1,Y2,Y3,Y4)
                                     ;  throw(error(type_error('Reify'(X4)),gecode_argument_error(dom(X0,X1,X2,X3,X4),arg=5))))
                                 ;  (is_int(X3,Y3)
                                     -> (is_int(X4,Y4)
                                         -> gecode_constraint_dom_166(Y0,Y1,Y2,Y3,Y4)
                                         ;  (is_Reify(X4,Y4)
                                             -> gecode_constraint_dom_168(Y0,Y1,Y2,Y3,Y4)
                                             ;  throw(error(type_error('Reify'(X4)),gecode_argument_error(dom(X0,X1,X2,X3,X4),arg=5)))))
                                     ;  throw(error(type_error(int(X3)),gecode_argument_error(dom(X0,X1,X2,X3,X4),arg=4)))))
                             ;  throw(error(type_error('SetRelType'(X2)),gecode_argument_error(dom(X0,X1,X2,X3,X4),arg=3))))
                         ;  throw(error(type_error('SetVar'(X1)),gecode_argument_error(dom(X0,X1,X2,X3,X4),arg=2)))))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(dom(X0,X1,X2,X3,X4),arg=1)))).

dom(X0,X1,X2,X3,X4,X5) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_SetVar(X1,Y1)
             -> (is_SetRelType(X2,Y2)
                 -> (is_int(X3,Y3)
                     -> (is_int(X4,Y4)
                         -> (is_Reify(X5,Y5)
                             -> gecode_constraint_dom_167(Y0,Y1,Y2,Y3,Y4,Y5)
                             ;  throw(error(type_error('Reify'(X5)),gecode_argument_error(dom(X0,X1,X2,X3,X4,X5),arg=6))))
                         ;  throw(error(type_error(int(X4)),gecode_argument_error(dom(X0,X1,X2,X3,X4,X5),arg=5))))
                     ;  throw(error(type_error(int(X3)),gecode_argument_error(dom(X0,X1,X2,X3,X4,X5),arg=4))))
                 ;  throw(error(type_error('SetRelType'(X2)),gecode_argument_error(dom(X0,X1,X2,X3,X4,X5),arg=3))))
             ;  throw(error(type_error('SetVar'(X1)),gecode_argument_error(dom(X0,X1,X2,X3,X4,X5),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(dom(X0,X1,X2,X3,X4,X5),arg=1)))).

element(X0,X1,X2,X3) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVarArgs(X1,Y1)
             -> (is_IntVar(X2,Y2)
                 -> (is_BoolVar(X3,Y3)
                     -> gecode_constraint_element_170(Y0,Y1,Y2,Y3)
                     ;  (is_int(X3,Y3)
                         -> gecode_constraint_element_171(Y0,Y1,Y2,Y3)
                         ;  throw(error(type_error(int(X3)),gecode_argument_error(element(X0,X1,X2,X3),arg=4)))))
                 ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(element(X0,X1,X2,X3),arg=3))))
             ;  (is_IntVarArgs(X1,Y1)
                 -> (is_IntVar(X2,Y2)
                     -> (is_int(X3,Y3)
                         -> gecode_constraint_element_173(Y0,Y1,Y2,Y3)
                         ;  (is_IntVar(X3,Y3)
                             -> gecode_constraint_element_175(Y0,Y1,Y2,Y3)
                             ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(element(X0,X1,X2,X3),arg=4)))))
                     ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(element(X0,X1,X2,X3),arg=3))))
                 ;  (is_IntArgs(X1,Y1)
                     -> (is_IntVar(X2,Y2)
                         -> (is_BoolVar(X3,Y3)
                             -> gecode_constraint_element_176(Y0,Y1,Y2,Y3)
                             ;  (is_int(X3,Y3)
                                 -> gecode_constraint_element_177(Y0,Y1,Y2,Y3)
                                 ;  (is_IntVar(X3,Y3)
                                     -> gecode_constraint_element_180(Y0,Y1,Y2,Y3)
                                     ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(element(X0,X1,X2,X3),arg=4))))))
                         ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(element(X0,X1,X2,X3),arg=3))))
                     ;  throw(error(type_error('IntArgs'(X1)),gecode_argument_error(element(X0,X1,X2,X3),arg=2))))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(element(X0,X1,X2,X3),arg=1)))).

element(X0,X1,X2,X3,X4,X5,X6) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVarArgs(X1,Y1)
             -> (is_IntVar(X2,Y2)
                 -> (is_int(X3,Y3)
                     -> (is_IntVar(X4,Y4)
                         -> (is_int(X5,Y5)
                             -> (is_BoolVar(X6,Y6)
                                 -> gecode_constraint_element_172(Y0,Y1,Y2,Y3,Y4,Y5,Y6)
                                 ;  throw(error(type_error('BoolVar'(X6)),gecode_argument_error(element(X0,X1,X2,X3,X4,X5,X6),arg=7))))
                             ;  throw(error(type_error(int(X5)),gecode_argument_error(element(X0,X1,X2,X3,X4,X5,X6),arg=6))))
                         ;  throw(error(type_error('IntVar'(X4)),gecode_argument_error(element(X0,X1,X2,X3,X4,X5,X6),arg=5))))
                     ;  throw(error(type_error(int(X3)),gecode_argument_error(element(X0,X1,X2,X3,X4,X5,X6),arg=4))))
                 ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(element(X0,X1,X2,X3,X4,X5,X6),arg=3))))
             ;  (is_IntVarArgs(X1,Y1)
                 -> (is_IntVar(X2,Y2)
                     -> (is_int(X3,Y3)
                         -> (is_IntVar(X4,Y4)
                             -> (is_int(X5,Y5)
                                 -> (is_IntVar(X6,Y6)
                                     -> gecode_constraint_element_174(Y0,Y1,Y2,Y3,Y4,Y5,Y6)
                                     ;  throw(error(type_error('IntVar'(X6)),gecode_argument_error(element(X0,X1,X2,X3,X4,X5,X6),arg=7))))
                                 ;  throw(error(type_error(int(X5)),gecode_argument_error(element(X0,X1,X2,X3,X4,X5,X6),arg=6))))
                             ;  throw(error(type_error('IntVar'(X4)),gecode_argument_error(element(X0,X1,X2,X3,X4,X5,X6),arg=5))))
                         ;  throw(error(type_error(int(X3)),gecode_argument_error(element(X0,X1,X2,X3,X4,X5,X6),arg=4))))
                     ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(element(X0,X1,X2,X3,X4,X5,X6),arg=3))))
                 ;  (is_IntArgs(X1,Y1)
                     -> (is_IntVar(X2,Y2)
                         -> (is_int(X3,Y3)
                             -> (is_IntVar(X4,Y4)
                                 -> (is_int(X5,Y5)
                                     -> (is_BoolVar(X6,Y6)
                                         -> gecode_constraint_element_178(Y0,Y1,Y2,Y3,Y4,Y5,Y6)
                                         ;  (is_IntVar(X6,Y6)
                                             -> gecode_constraint_element_179(Y0,Y1,Y2,Y3,Y4,Y5,Y6)
                                             ;  throw(error(type_error('IntVar'(X6)),gecode_argument_error(element(X0,X1,X2,X3,X4,X5,X6),arg=7)))))
                                     ;  throw(error(type_error(int(X5)),gecode_argument_error(element(X0,X1,X2,X3,X4,X5,X6),arg=6))))
                                 ;  throw(error(type_error('IntVar'(X4)),gecode_argument_error(element(X0,X1,X2,X3,X4,X5,X6),arg=5))))
                             ;  throw(error(type_error(int(X3)),gecode_argument_error(element(X0,X1,X2,X3,X4,X5,X6),arg=4))))
                         ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(element(X0,X1,X2,X3,X4,X5,X6),arg=3))))
                     ;  throw(error(type_error('IntArgs'(X1)),gecode_argument_error(element(X0,X1,X2,X3,X4,X5,X6),arg=2))))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(element(X0,X1,X2,X3,X4,X5,X6),arg=1)))).

extensional(X0,X1,X2,X3) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVarArgs(X1,Y1)
             -> (is_TupleSet(X2,Y2)
                 -> (is_bool(X3,Y3)
                     -> gecode_constraint_extensional_181(Y0,Y1,Y2,Y3)
                     ;  throw(error(type_error(bool(X3)),gecode_argument_error(extensional(X0,X1,X2,X3),arg=4))))
                 ;  throw(error(type_error('TupleSet'(X2)),gecode_argument_error(extensional(X0,X1,X2,X3),arg=3))))
             ;  (is_IntVarArgs(X1,Y1)
                 -> (is_TupleSet(X2,Y2)
                     -> (is_bool(X3,Y3)
                         -> gecode_constraint_extensional_184(Y0,Y1,Y2,Y3)
                         ;  throw(error(type_error(bool(X3)),gecode_argument_error(extensional(X0,X1,X2,X3),arg=4))))
                     ;  throw(error(type_error('TupleSet'(X2)),gecode_argument_error(extensional(X0,X1,X2,X3),arg=3))))
                 ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(extensional(X0,X1,X2,X3),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(extensional(X0,X1,X2,X3),arg=1)))).

extensional(X0,X1,X2,X3,X4) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVarArgs(X1,Y1)
             -> (is_TupleSet(X2,Y2)
                 -> (is_bool(X3,Y3)
                     -> (is_Reify(X4,Y4)
                         -> gecode_constraint_extensional_182(Y0,Y1,Y2,Y3,Y4)
                         ;  throw(error(type_error('Reify'(X4)),gecode_argument_error(extensional(X0,X1,X2,X3,X4),arg=5))))
                     ;  throw(error(type_error(bool(X3)),gecode_argument_error(extensional(X0,X1,X2,X3,X4),arg=4))))
                 ;  throw(error(type_error('TupleSet'(X2)),gecode_argument_error(extensional(X0,X1,X2,X3,X4),arg=3))))
             ;  (is_IntVarArgs(X1,Y1)
                 -> (is_TupleSet(X2,Y2)
                     -> (is_bool(X3,Y3)
                         -> (is_Reify(X4,Y4)
                             -> gecode_constraint_extensional_185(Y0,Y1,Y2,Y3,Y4)
                             ;  throw(error(type_error('Reify'(X4)),gecode_argument_error(extensional(X0,X1,X2,X3,X4),arg=5))))
                         ;  throw(error(type_error(bool(X3)),gecode_argument_error(extensional(X0,X1,X2,X3,X4),arg=4))))
                     ;  throw(error(type_error('TupleSet'(X2)),gecode_argument_error(extensional(X0,X1,X2,X3,X4),arg=3))))
                 ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(extensional(X0,X1,X2,X3,X4),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(extensional(X0,X1,X2,X3,X4),arg=1)))).

extensional(X0,X1,X2) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVarArgs(X1,Y1)
             -> (is_DFA(X2,Y2)
                 -> gecode_constraint_extensional_183(Y0,Y1,Y2)
                 ;  throw(error(type_error('DFA'(X2)),gecode_argument_error(extensional(X0,X1,X2),arg=3))))
             ;  (is_IntVarArgs(X1,Y1)
                 -> (is_DFA(X2,Y2)
                     -> gecode_constraint_extensional_186(Y0,Y1,Y2)
                     ;  throw(error(type_error('DFA'(X2)),gecode_argument_error(extensional(X0,X1,X2),arg=3))))
                 ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(extensional(X0,X1,X2),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(extensional(X0,X1,X2),arg=1)))).

ite(X0,X1,X2,X3,X4) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVar(X1,Y1)
             -> (is_BoolVar(X2,Y2)
                 -> (is_BoolVar(X3,Y3)
                     -> (is_BoolVar(X4,Y4)
                         -> gecode_constraint_ite_187(Y0,Y1,Y2,Y3,Y4)
                         ;  throw(error(type_error('BoolVar'(X4)),gecode_argument_error(ite(X0,X1,X2,X3,X4),arg=5))))
                     ;  throw(error(type_error('BoolVar'(X3)),gecode_argument_error(ite(X0,X1,X2,X3,X4),arg=4))))
                 ;  (is_FloatVar(X2,Y2)
                     -> (is_FloatVar(X3,Y3)
                         -> (is_FloatVar(X4,Y4)
                             -> gecode_constraint_ite_188(Y0,Y1,Y2,Y3,Y4)
                             ;  throw(error(type_error('FloatVar'(X4)),gecode_argument_error(ite(X0,X1,X2,X3,X4),arg=5))))
                         ;  throw(error(type_error('FloatVar'(X3)),gecode_argument_error(ite(X0,X1,X2,X3,X4),arg=4))))
                     ;  (is_IntVar(X2,Y2)
                         -> (is_IntVar(X3,Y3)
                             -> (is_IntVar(X4,Y4)
                                 -> gecode_constraint_ite_189(Y0,Y1,Y2,Y3,Y4)
                                 ;  throw(error(type_error('IntVar'(X4)),gecode_argument_error(ite(X0,X1,X2,X3,X4),arg=5))))
                             ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(ite(X0,X1,X2,X3,X4),arg=4))))
                         ;  (is_SetVar(X2,Y2)
                             -> (is_SetVar(X3,Y3)
                                 -> (is_SetVar(X4,Y4)
                                     -> gecode_constraint_ite_190(Y0,Y1,Y2,Y3,Y4)
                                     ;  throw(error(type_error('SetVar'(X4)),gecode_argument_error(ite(X0,X1,X2,X3,X4),arg=5))))
                                 ;  throw(error(type_error('SetVar'(X3)),gecode_argument_error(ite(X0,X1,X2,X3,X4),arg=4))))
                             ;  throw(error(type_error('SetVar'(X2)),gecode_argument_error(ite(X0,X1,X2,X3,X4),arg=3)))))))
             ;  throw(error(type_error('BoolVar'(X1)),gecode_argument_error(ite(X0,X1,X2,X3,X4),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(ite(X0,X1,X2,X3,X4),arg=1)))).

linear(X0,X1,X2,X3) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVarArgs(X1,Y1)
             -> (is_IntRelType(X2,Y2)
                 -> (is_int(X3,Y3)
                     -> gecode_constraint_linear_191(Y0,Y1,Y2,Y3)
                     ;  (is_IntVar(X3,Y3)
                         -> gecode_constraint_linear_193(Y0,Y1,Y2,Y3)
                         ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(linear(X0,X1,X2,X3),arg=4)))))
                 ;  throw(error(type_error('IntRelType'(X2)),gecode_argument_error(linear(X0,X1,X2,X3),arg=3))))
             ;  (is_FloatVarArgs(X1,Y1)
                 -> (is_FloatRelType(X2,Y2)
                     -> (is_FloatVal(X3,Y3)
                         -> gecode_constraint_linear_199(Y0,Y1,Y2,Y3)
                         ;  (is_FloatVar(X3,Y3)
                             -> gecode_constraint_linear_201(Y0,Y1,Y2,Y3)
                             ;  throw(error(type_error('FloatVar'(X3)),gecode_argument_error(linear(X0,X1,X2,X3),arg=4)))))
                     ;  throw(error(type_error('FloatRelType'(X2)),gecode_argument_error(linear(X0,X1,X2,X3),arg=3))))
                 ;  (is_IntVarArgs(X1,Y1)
                     -> (is_IntRelType(X2,Y2)
                         -> (is_int(X3,Y3)
                             -> gecode_constraint_linear_211(Y0,Y1,Y2,Y3)
                             ;  (is_IntVar(X3,Y3)
                                 -> gecode_constraint_linear_213(Y0,Y1,Y2,Y3)
                                 ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(linear(X0,X1,X2,X3),arg=4)))))
                         ;  throw(error(type_error('IntRelType'(X2)),gecode_argument_error(linear(X0,X1,X2,X3),arg=3))))
                     ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(linear(X0,X1,X2,X3),arg=2))))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(linear(X0,X1,X2,X3),arg=1)))).

linear(X0,X1,X2,X3,X4) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVarArgs(X1,Y1)
             -> (is_IntRelType(X2,Y2)
                 -> (is_int(X3,Y3)
                     -> (is_Reify(X4,Y4)
                         -> gecode_constraint_linear_192(Y0,Y1,Y2,Y3,Y4)
                         ;  throw(error(type_error('Reify'(X4)),gecode_argument_error(linear(X0,X1,X2,X3,X4),arg=5))))
                     ;  (is_IntVar(X3,Y3)
                         -> (is_Reify(X4,Y4)
                             -> gecode_constraint_linear_194(Y0,Y1,Y2,Y3,Y4)
                             ;  throw(error(type_error('Reify'(X4)),gecode_argument_error(linear(X0,X1,X2,X3,X4),arg=5))))
                         ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(linear(X0,X1,X2,X3,X4),arg=4)))))
                 ;  throw(error(type_error('IntRelType'(X2)),gecode_argument_error(linear(X0,X1,X2,X3,X4),arg=3))))
             ;  (is_FloatValArgs(X1,Y1)
                 -> (is_FloatVarArgs(X2,Y2)
                     -> (is_FloatRelType(X3,Y3)
                         -> (is_FloatVal(X4,Y4)
                             -> gecode_constraint_linear_195(Y0,Y1,Y2,Y3,Y4)
                             ;  (is_FloatVar(X4,Y4)
                                 -> gecode_constraint_linear_197(Y0,Y1,Y2,Y3,Y4)
                                 ;  throw(error(type_error('FloatVar'(X4)),gecode_argument_error(linear(X0,X1,X2,X3,X4),arg=5)))))
                         ;  throw(error(type_error('FloatRelType'(X3)),gecode_argument_error(linear(X0,X1,X2,X3,X4),arg=4))))
                     ;  throw(error(type_error('FloatVarArgs'(X2)),gecode_argument_error(linear(X0,X1,X2,X3,X4),arg=3))))
                 ;  (is_FloatVarArgs(X1,Y1)
                     -> (is_FloatRelType(X2,Y2)
                         -> (is_FloatVal(X3,Y3)
                             -> (is_Reify(X4,Y4)
                                 -> gecode_constraint_linear_200(Y0,Y1,Y2,Y3,Y4)
                                 ;  throw(error(type_error('Reify'(X4)),gecode_argument_error(linear(X0,X1,X2,X3,X4),arg=5))))
                             ;  (is_FloatVar(X3,Y3)
                                 -> (is_Reify(X4,Y4)
                                     -> gecode_constraint_linear_202(Y0,Y1,Y2,Y3,Y4)
                                     ;  throw(error(type_error('Reify'(X4)),gecode_argument_error(linear(X0,X1,X2,X3,X4),arg=5))))
                                 ;  throw(error(type_error('FloatVar'(X3)),gecode_argument_error(linear(X0,X1,X2,X3,X4),arg=4)))))
                         ;  throw(error(type_error('FloatRelType'(X2)),gecode_argument_error(linear(X0,X1,X2,X3,X4),arg=3))))
                     ;  (is_IntArgs(X1,Y1)
                         -> (is_BoolVarArgs(X2,Y2)
                             -> (is_IntRelType(X3,Y3)
                                 -> (is_int(X4,Y4)
                                     -> gecode_constraint_linear_203(Y0,Y1,Y2,Y3,Y4)
                                     ;  (is_IntVar(X4,Y4)
                                         -> gecode_constraint_linear_205(Y0,Y1,Y2,Y3,Y4)
                                         ;  throw(error(type_error('IntVar'(X4)),gecode_argument_error(linear(X0,X1,X2,X3,X4),arg=5)))))
                                 ;  throw(error(type_error('IntRelType'(X3)),gecode_argument_error(linear(X0,X1,X2,X3,X4),arg=4))))
                             ;  (is_IntVarArgs(X2,Y2)
                                 -> (is_IntRelType(X3,Y3)
                                     -> (is_int(X4,Y4)
                                         -> gecode_constraint_linear_207(Y0,Y1,Y2,Y3,Y4)
                                         ;  (is_IntVar(X4,Y4)
                                             -> gecode_constraint_linear_209(Y0,Y1,Y2,Y3,Y4)
                                             ;  throw(error(type_error('IntVar'(X4)),gecode_argument_error(linear(X0,X1,X2,X3,X4),arg=5)))))
                                     ;  throw(error(type_error('IntRelType'(X3)),gecode_argument_error(linear(X0,X1,X2,X3,X4),arg=4))))
                                 ;  throw(error(type_error('IntVarArgs'(X2)),gecode_argument_error(linear(X0,X1,X2,X3,X4),arg=3)))))
                         ;  (is_IntVarArgs(X1,Y1)
                             -> (is_IntRelType(X2,Y2)
                                 -> (is_int(X3,Y3)
                                     -> (is_Reify(X4,Y4)
                                         -> gecode_constraint_linear_212(Y0,Y1,Y2,Y3,Y4)
                                         ;  throw(error(type_error('Reify'(X4)),gecode_argument_error(linear(X0,X1,X2,X3,X4),arg=5))))
                                     ;  (is_IntVar(X3,Y3)
                                         -> (is_Reify(X4,Y4)
                                             -> gecode_constraint_linear_214(Y0,Y1,Y2,Y3,Y4)
                                             ;  throw(error(type_error('Reify'(X4)),gecode_argument_error(linear(X0,X1,X2,X3,X4),arg=5))))
                                         ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(linear(X0,X1,X2,X3,X4),arg=4)))))
                                 ;  throw(error(type_error('IntRelType'(X2)),gecode_argument_error(linear(X0,X1,X2,X3,X4),arg=3))))
                             ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(linear(X0,X1,X2,X3,X4),arg=2))))))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(linear(X0,X1,X2,X3,X4),arg=1)))).

linear(X0,X1,X2,X3,X4,X5) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_FloatValArgs(X1,Y1)
             -> (is_FloatVarArgs(X2,Y2)
                 -> (is_FloatRelType(X3,Y3)
                     -> (is_FloatVal(X4,Y4)
                         -> (is_Reify(X5,Y5)
                             -> gecode_constraint_linear_196(Y0,Y1,Y2,Y3,Y4,Y5)
                             ;  throw(error(type_error('Reify'(X5)),gecode_argument_error(linear(X0,X1,X2,X3,X4,X5),arg=6))))
                         ;  (is_FloatVar(X4,Y4)
                             -> (is_Reify(X5,Y5)
                                 -> gecode_constraint_linear_198(Y0,Y1,Y2,Y3,Y4,Y5)
                                 ;  throw(error(type_error('Reify'(X5)),gecode_argument_error(linear(X0,X1,X2,X3,X4,X5),arg=6))))
                             ;  throw(error(type_error('FloatVar'(X4)),gecode_argument_error(linear(X0,X1,X2,X3,X4,X5),arg=5)))))
                     ;  throw(error(type_error('FloatRelType'(X3)),gecode_argument_error(linear(X0,X1,X2,X3,X4,X5),arg=4))))
                 ;  throw(error(type_error('FloatVarArgs'(X2)),gecode_argument_error(linear(X0,X1,X2,X3,X4,X5),arg=3))))
             ;  (is_IntArgs(X1,Y1)
                 -> (is_BoolVarArgs(X2,Y2)
                     -> (is_IntRelType(X3,Y3)
                         -> (is_int(X4,Y4)
                             -> (is_Reify(X5,Y5)
                                 -> gecode_constraint_linear_204(Y0,Y1,Y2,Y3,Y4,Y5)
                                 ;  throw(error(type_error('Reify'(X5)),gecode_argument_error(linear(X0,X1,X2,X3,X4,X5),arg=6))))
                             ;  (is_IntVar(X4,Y4)
                                 -> (is_Reify(X5,Y5)
                                     -> gecode_constraint_linear_206(Y0,Y1,Y2,Y3,Y4,Y5)
                                     ;  throw(error(type_error('Reify'(X5)),gecode_argument_error(linear(X0,X1,X2,X3,X4,X5),arg=6))))
                                 ;  throw(error(type_error('IntVar'(X4)),gecode_argument_error(linear(X0,X1,X2,X3,X4,X5),arg=5)))))
                         ;  throw(error(type_error('IntRelType'(X3)),gecode_argument_error(linear(X0,X1,X2,X3,X4,X5),arg=4))))
                     ;  (is_IntVarArgs(X2,Y2)
                         -> (is_IntRelType(X3,Y3)
                             -> (is_int(X4,Y4)
                                 -> (is_Reify(X5,Y5)
                                     -> gecode_constraint_linear_208(Y0,Y1,Y2,Y3,Y4,Y5)
                                     ;  throw(error(type_error('Reify'(X5)),gecode_argument_error(linear(X0,X1,X2,X3,X4,X5),arg=6))))
                                 ;  (is_IntVar(X4,Y4)
                                     -> (is_Reify(X5,Y5)
                                         -> gecode_constraint_linear_210(Y0,Y1,Y2,Y3,Y4,Y5)
                                         ;  throw(error(type_error('Reify'(X5)),gecode_argument_error(linear(X0,X1,X2,X3,X4,X5),arg=6))))
                                     ;  throw(error(type_error('IntVar'(X4)),gecode_argument_error(linear(X0,X1,X2,X3,X4,X5),arg=5)))))
                             ;  throw(error(type_error('IntRelType'(X3)),gecode_argument_error(linear(X0,X1,X2,X3,X4,X5),arg=4))))
                         ;  throw(error(type_error('IntVarArgs'(X2)),gecode_argument_error(linear(X0,X1,X2,X3,X4,X5),arg=3)))))
                 ;  throw(error(type_error('IntArgs'(X1)),gecode_argument_error(linear(X0,X1,X2,X3,X4,X5),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(linear(X0,X1,X2,X3,X4,X5),arg=1)))).

max(X0,X1,X2) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_FloatVarArgs(X1,Y1)
             -> (is_FloatVar(X2,Y2)
                 -> gecode_constraint_max_215(Y0,Y1,Y2)
                 ;  throw(error(type_error('FloatVar'(X2)),gecode_argument_error(max(X0,X1,X2),arg=3))))
             ;  (is_IntVarArgs(X1,Y1)
                 -> (is_IntVar(X2,Y2)
                     -> gecode_constraint_max_216(Y0,Y1,Y2)
                     ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(max(X0,X1,X2),arg=3))))
                 ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(max(X0,X1,X2),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(max(X0,X1,X2),arg=1)))).

max(X0,X1,X2,X3) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_FloatVar(X1,Y1)
             -> (is_FloatVar(X2,Y2)
                 -> (is_FloatVar(X3,Y3)
                     -> gecode_constraint_max_217(Y0,Y1,Y2,Y3)
                     ;  throw(error(type_error('FloatVar'(X3)),gecode_argument_error(max(X0,X1,X2,X3),arg=4))))
                 ;  throw(error(type_error('FloatVar'(X2)),gecode_argument_error(max(X0,X1,X2,X3),arg=3))))
             ;  (is_IntVar(X1,Y1)
                 -> (is_IntVar(X2,Y2)
                     -> (is_IntVar(X3,Y3)
                         -> gecode_constraint_max_218(Y0,Y1,Y2,Y3)
                         ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(max(X0,X1,X2,X3),arg=4))))
                     ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(max(X0,X1,X2,X3),arg=3))))
                 ;  throw(error(type_error('IntVar'(X1)),gecode_argument_error(max(X0,X1,X2,X3),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(max(X0,X1,X2,X3),arg=1)))).

member(X0,X1,X2) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVarArgs(X1,Y1)
             -> (is_BoolVar(X2,Y2)
                 -> gecode_constraint_member_219(Y0,Y1,Y2)
                 ;  throw(error(type_error('BoolVar'(X2)),gecode_argument_error(member(X0,X1,X2),arg=3))))
             ;  (is_IntVarArgs(X1,Y1)
                 -> (is_IntVar(X2,Y2)
                     -> gecode_constraint_member_221(Y0,Y1,Y2)
                     ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(member(X0,X1,X2),arg=3))))
                 ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(member(X0,X1,X2),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(member(X0,X1,X2),arg=1)))).

member(X0,X1,X2,X3) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVarArgs(X1,Y1)
             -> (is_BoolVar(X2,Y2)
                 -> (is_Reify(X3,Y3)
                     -> gecode_constraint_member_220(Y0,Y1,Y2,Y3)
                     ;  throw(error(type_error('Reify'(X3)),gecode_argument_error(member(X0,X1,X2,X3),arg=4))))
                 ;  throw(error(type_error('BoolVar'(X2)),gecode_argument_error(member(X0,X1,X2,X3),arg=3))))
             ;  (is_IntVarArgs(X1,Y1)
                 -> (is_IntVar(X2,Y2)
                     -> (is_Reify(X3,Y3)
                         -> gecode_constraint_member_222(Y0,Y1,Y2,Y3)
                         ;  throw(error(type_error('Reify'(X3)),gecode_argument_error(member(X0,X1,X2,X3),arg=4))))
                     ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(member(X0,X1,X2,X3),arg=3))))
                 ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(member(X0,X1,X2,X3),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(member(X0,X1,X2,X3),arg=1)))).

min(X0,X1,X2) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_FloatVarArgs(X1,Y1)
             -> (is_FloatVar(X2,Y2)
                 -> gecode_constraint_min_223(Y0,Y1,Y2)
                 ;  throw(error(type_error('FloatVar'(X2)),gecode_argument_error(min(X0,X1,X2),arg=3))))
             ;  (is_IntVarArgs(X1,Y1)
                 -> (is_IntVar(X2,Y2)
                     -> gecode_constraint_min_224(Y0,Y1,Y2)
                     ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(min(X0,X1,X2),arg=3))))
                 ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(min(X0,X1,X2),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(min(X0,X1,X2),arg=1)))).

min(X0,X1,X2,X3) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_FloatVar(X1,Y1)
             -> (is_FloatVar(X2,Y2)
                 -> (is_FloatVar(X3,Y3)
                     -> gecode_constraint_min_225(Y0,Y1,Y2,Y3)
                     ;  throw(error(type_error('FloatVar'(X3)),gecode_argument_error(min(X0,X1,X2,X3),arg=4))))
                 ;  throw(error(type_error('FloatVar'(X2)),gecode_argument_error(min(X0,X1,X2,X3),arg=3))))
             ;  (is_IntVar(X1,Y1)
                 -> (is_IntVar(X2,Y2)
                     -> (is_IntVar(X3,Y3)
                         -> gecode_constraint_min_226(Y0,Y1,Y2,Y3)
                         ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(min(X0,X1,X2,X3),arg=4))))
                     ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(min(X0,X1,X2,X3),arg=3))))
                 ;  throw(error(type_error('IntVar'(X1)),gecode_argument_error(min(X0,X1,X2,X3),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(min(X0,X1,X2,X3),arg=1)))).

mod(X0,X1,X2,X3) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_IntVar(X1,Y1)
             -> (is_IntVar(X2,Y2)
                 -> (is_IntVar(X3,Y3)
                     -> gecode_constraint_mod_227(Y0,Y1,Y2,Y3)
                     ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(mod(X0,X1,X2,X3),arg=4))))
                 ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(mod(X0,X1,X2,X3),arg=3))))
             ;  throw(error(type_error('IntVar'(X1)),gecode_argument_error(mod(X0,X1,X2,X3),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(mod(X0,X1,X2,X3),arg=1)))).

mult(X0,X1,X2,X3) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_FloatVar(X1,Y1)
             -> (is_FloatVar(X2,Y2)
                 -> (is_FloatVar(X3,Y3)
                     -> gecode_constraint_mult_228(Y0,Y1,Y2,Y3)
                     ;  throw(error(type_error('FloatVar'(X3)),gecode_argument_error(mult(X0,X1,X2,X3),arg=4))))
                 ;  throw(error(type_error('FloatVar'(X2)),gecode_argument_error(mult(X0,X1,X2,X3),arg=3))))
             ;  (is_IntVar(X1,Y1)
                 -> (is_IntVar(X2,Y2)
                     -> (is_IntVar(X3,Y3)
                         -> gecode_constraint_mult_229(Y0,Y1,Y2,Y3)
                         ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(mult(X0,X1,X2,X3),arg=4))))
                     ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(mult(X0,X1,X2,X3),arg=3))))
                 ;  throw(error(type_error('IntVar'(X1)),gecode_argument_error(mult(X0,X1,X2,X3),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(mult(X0,X1,X2,X3),arg=1)))).

nooverlap(X0,X1,X2,X3,X4,X5) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_IntVarArgs(X1,Y1)
             -> (is_IntArgs(X2,Y2)
                 -> (is_IntVarArgs(X3,Y3)
                     -> (is_IntArgs(X4,Y4)
                         -> (is_BoolVarArgs(X5,Y5)
                             -> gecode_constraint_nooverlap_230(Y0,Y1,Y2,Y3,Y4,Y5)
                             ;  throw(error(type_error('BoolVarArgs'(X5)),gecode_argument_error(nooverlap(X0,X1,X2,X3,X4,X5),arg=6))))
                         ;  throw(error(type_error('IntArgs'(X4)),gecode_argument_error(nooverlap(X0,X1,X2,X3,X4,X5),arg=5))))
                     ;  throw(error(type_error('IntVarArgs'(X3)),gecode_argument_error(nooverlap(X0,X1,X2,X3,X4,X5),arg=4))))
                 ;  throw(error(type_error('IntArgs'(X2)),gecode_argument_error(nooverlap(X0,X1,X2,X3,X4,X5),arg=3))))
             ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(nooverlap(X0,X1,X2,X3,X4,X5),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(nooverlap(X0,X1,X2,X3,X4,X5),arg=1)))).

nooverlap(X0,X1,X2,X3,X4) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_IntVarArgs(X1,Y1)
             -> (is_IntArgs(X2,Y2)
                 -> (is_IntVarArgs(X3,Y3)
                     -> (is_IntArgs(X4,Y4)
                         -> gecode_constraint_nooverlap_231(Y0,Y1,Y2,Y3,Y4)
                         ;  throw(error(type_error('IntArgs'(X4)),gecode_argument_error(nooverlap(X0,X1,X2,X3,X4),arg=5))))
                     ;  throw(error(type_error('IntVarArgs'(X3)),gecode_argument_error(nooverlap(X0,X1,X2,X3,X4),arg=4))))
                 ;  throw(error(type_error('IntArgs'(X2)),gecode_argument_error(nooverlap(X0,X1,X2,X3,X4),arg=3))))
             ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(nooverlap(X0,X1,X2,X3,X4),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(nooverlap(X0,X1,X2,X3,X4),arg=1)))).

nooverlap(X0,X1,X2,X3,X4,X5,X6,X7) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_IntVarArgs(X1,Y1)
             -> (is_IntVarArgs(X2,Y2)
                 -> (is_IntVarArgs(X3,Y3)
                     -> (is_IntVarArgs(X4,Y4)
                         -> (is_IntVarArgs(X5,Y5)
                             -> (is_IntVarArgs(X6,Y6)
                                 -> (is_BoolVarArgs(X7,Y7)
                                     -> gecode_constraint_nooverlap_232(Y0,Y1,Y2,Y3,Y4,Y5,Y6,Y7)
                                     ;  throw(error(type_error('BoolVarArgs'(X7)),gecode_argument_error(nooverlap(X0,X1,X2,X3,X4,X5,X6,X7),arg=8))))
                                 ;  throw(error(type_error('IntVarArgs'(X6)),gecode_argument_error(nooverlap(X0,X1,X2,X3,X4,X5,X6,X7),arg=7))))
                             ;  throw(error(type_error('IntVarArgs'(X5)),gecode_argument_error(nooverlap(X0,X1,X2,X3,X4,X5,X6,X7),arg=6))))
                         ;  throw(error(type_error('IntVarArgs'(X4)),gecode_argument_error(nooverlap(X0,X1,X2,X3,X4,X5,X6,X7),arg=5))))
                     ;  throw(error(type_error('IntVarArgs'(X3)),gecode_argument_error(nooverlap(X0,X1,X2,X3,X4,X5,X6,X7),arg=4))))
                 ;  throw(error(type_error('IntVarArgs'(X2)),gecode_argument_error(nooverlap(X0,X1,X2,X3,X4,X5,X6,X7),arg=3))))
             ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(nooverlap(X0,X1,X2,X3,X4,X5,X6,X7),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(nooverlap(X0,X1,X2,X3,X4,X5,X6,X7),arg=1)))).

nooverlap(X0,X1,X2,X3,X4,X5,X6) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_IntVarArgs(X1,Y1)
             -> (is_IntVarArgs(X2,Y2)
                 -> (is_IntVarArgs(X3,Y3)
                     -> (is_IntVarArgs(X4,Y4)
                         -> (is_IntVarArgs(X5,Y5)
                             -> (is_IntVarArgs(X6,Y6)
                                 -> gecode_constraint_nooverlap_233(Y0,Y1,Y2,Y3,Y4,Y5,Y6)
                                 ;  throw(error(type_error('IntVarArgs'(X6)),gecode_argument_error(nooverlap(X0,X1,X2,X3,X4,X5,X6),arg=7))))
                             ;  throw(error(type_error('IntVarArgs'(X5)),gecode_argument_error(nooverlap(X0,X1,X2,X3,X4,X5,X6),arg=6))))
                         ;  throw(error(type_error('IntVarArgs'(X4)),gecode_argument_error(nooverlap(X0,X1,X2,X3,X4,X5,X6),arg=5))))
                     ;  throw(error(type_error('IntVarArgs'(X3)),gecode_argument_error(nooverlap(X0,X1,X2,X3,X4,X5,X6),arg=4))))
                 ;  throw(error(type_error('IntVarArgs'(X2)),gecode_argument_error(nooverlap(X0,X1,X2,X3,X4,X5,X6),arg=3))))
             ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(nooverlap(X0,X1,X2,X3,X4,X5,X6),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(nooverlap(X0,X1,X2,X3,X4,X5,X6),arg=1)))).

nroot(X0,X1,X2,X3) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_FloatVar(X1,Y1)
             -> (is_int(X2,Y2)
                 -> (is_FloatVar(X3,Y3)
                     -> gecode_constraint_nroot_234(Y0,Y1,Y2,Y3)
                     ;  throw(error(type_error('FloatVar'(X3)),gecode_argument_error(nroot(X0,X1,X2,X3),arg=4))))
                 ;  throw(error(type_error(int(X2)),gecode_argument_error(nroot(X0,X1,X2,X3),arg=3))))
             ;  (is_IntVar(X1,Y1)
                 -> (is_int(X2,Y2)
                     -> (is_IntVar(X3,Y3)
                         -> gecode_constraint_nroot_235(Y0,Y1,Y2,Y3)
                         ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(nroot(X0,X1,X2,X3),arg=4))))
                     ;  throw(error(type_error(int(X2)),gecode_argument_error(nroot(X0,X1,X2,X3),arg=3))))
                 ;  throw(error(type_error('IntVar'(X1)),gecode_argument_error(nroot(X0,X1,X2,X3),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(nroot(X0,X1,X2,X3),arg=1)))).

nvalues(X0,X1,X2,X3) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVarArgs(X1,Y1)
             -> (is_IntRelType(X2,Y2)
                 -> (is_int(X3,Y3)
                     -> gecode_constraint_nvalues_236(Y0,Y1,Y2,Y3)
                     ;  (is_IntVar(X3,Y3)
                         -> gecode_constraint_nvalues_237(Y0,Y1,Y2,Y3)
                         ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(nvalues(X0,X1,X2,X3),arg=4)))))
                 ;  throw(error(type_error('IntRelType'(X2)),gecode_argument_error(nvalues(X0,X1,X2,X3),arg=3))))
             ;  (is_IntVarArgs(X1,Y1)
                 -> (is_IntRelType(X2,Y2)
                     -> (is_int(X3,Y3)
                         -> gecode_constraint_nvalues_238(Y0,Y1,Y2,Y3)
                         ;  (is_IntVar(X3,Y3)
                             -> gecode_constraint_nvalues_239(Y0,Y1,Y2,Y3)
                             ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(nvalues(X0,X1,X2,X3),arg=4)))))
                     ;  throw(error(type_error('IntRelType'(X2)),gecode_argument_error(nvalues(X0,X1,X2,X3),arg=3))))
                 ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(nvalues(X0,X1,X2,X3),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(nvalues(X0,X1,X2,X3),arg=1)))).

order(X0,X1,X2,X3,X4,X5) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_IntVar(X1,Y1)
             -> (is_int(X2,Y2)
                 -> (is_IntVar(X3,Y3)
                     -> (is_int(X4,Y4)
                         -> (is_BoolVar(X5,Y5)
                             -> gecode_constraint_order_240(Y0,Y1,Y2,Y3,Y4,Y5)
                             ;  throw(error(type_error('BoolVar'(X5)),gecode_argument_error(order(X0,X1,X2,X3,X4,X5),arg=6))))
                         ;  throw(error(type_error(int(X4)),gecode_argument_error(order(X0,X1,X2,X3,X4,X5),arg=5))))
                     ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(order(X0,X1,X2,X3,X4,X5),arg=4))))
                 ;  throw(error(type_error(int(X2)),gecode_argument_error(order(X0,X1,X2,X3,X4,X5),arg=3))))
             ;  throw(error(type_error('IntVar'(X1)),gecode_argument_error(order(X0,X1,X2,X3,X4,X5),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(order(X0,X1,X2,X3,X4,X5),arg=1)))).

path(X0,X1,X2,X3,X4,X5,X6) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_IntArgs(X1,Y1)
             -> (is_IntVarArgs(X2,Y2)
                 -> (is_IntVar(X3,Y3)
                     -> (is_IntVar(X4,Y4)
                         -> (is_IntVarArgs(X5,Y5)
                             -> (is_IntVar(X6,Y6)
                                 -> gecode_constraint_path_241(Y0,Y1,Y2,Y3,Y4,Y5,Y6)
                                 ;  throw(error(type_error('IntVar'(X6)),gecode_argument_error(path(X0,X1,X2,X3,X4,X5,X6),arg=7))))
                             ;  throw(error(type_error('IntVarArgs'(X5)),gecode_argument_error(path(X0,X1,X2,X3,X4,X5,X6),arg=6))))
                         ;  throw(error(type_error('IntVar'(X4)),gecode_argument_error(path(X0,X1,X2,X3,X4,X5,X6),arg=5))))
                     ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(path(X0,X1,X2,X3,X4,X5,X6),arg=4))))
                 ;  (is_int(X2,Y2)
                     -> (is_IntVarArgs(X3,Y3)
                         -> (is_IntVar(X4,Y4)
                             -> (is_IntVar(X5,Y5)
                                 -> (is_IntVar(X6,Y6)
                                     -> gecode_constraint_path_244(Y0,Y1,Y2,Y3,Y4,Y5,Y6)
                                     ;  throw(error(type_error('IntVar'(X6)),gecode_argument_error(path(X0,X1,X2,X3,X4,X5,X6),arg=7))))
                                 ;  throw(error(type_error('IntVar'(X5)),gecode_argument_error(path(X0,X1,X2,X3,X4,X5,X6),arg=6))))
                             ;  throw(error(type_error('IntVar'(X4)),gecode_argument_error(path(X0,X1,X2,X3,X4,X5,X6),arg=5))))
                         ;  throw(error(type_error('IntVarArgs'(X3)),gecode_argument_error(path(X0,X1,X2,X3,X4,X5,X6),arg=4))))
                     ;  throw(error(type_error(int(X2)),gecode_argument_error(path(X0,X1,X2,X3,X4,X5,X6),arg=3)))))
             ;  throw(error(type_error('IntArgs'(X1)),gecode_argument_error(path(X0,X1,X2,X3,X4,X5,X6),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(path(X0,X1,X2,X3,X4,X5,X6),arg=1)))).

path(X0,X1,X2,X3,X4,X5) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_IntArgs(X1,Y1)
             -> (is_IntVarArgs(X2,Y2)
                 -> (is_IntVar(X3,Y3)
                     -> (is_IntVar(X4,Y4)
                         -> (is_IntVar(X5,Y5)
                             -> gecode_constraint_path_242(Y0,Y1,Y2,Y3,Y4,Y5)
                             ;  throw(error(type_error('IntVar'(X5)),gecode_argument_error(path(X0,X1,X2,X3,X4,X5),arg=6))))
                         ;  throw(error(type_error('IntVar'(X4)),gecode_argument_error(path(X0,X1,X2,X3,X4,X5),arg=5))))
                     ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(path(X0,X1,X2,X3,X4,X5),arg=4))))
                 ;  throw(error(type_error('IntVarArgs'(X2)),gecode_argument_error(path(X0,X1,X2,X3,X4,X5),arg=3))))
             ;  throw(error(type_error('IntArgs'(X1)),gecode_argument_error(path(X0,X1,X2,X3,X4,X5),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(path(X0,X1,X2,X3,X4,X5),arg=1)))).

path(X0,X1,X2,X3,X4,X5,X6,X7) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_IntArgs(X1,Y1)
             -> (is_int(X2,Y2)
                 -> (is_IntVarArgs(X3,Y3)
                     -> (is_IntVar(X4,Y4)
                         -> (is_IntVar(X5,Y5)
                             -> (is_IntVarArgs(X6,Y6)
                                 -> (is_IntVar(X7,Y7)
                                     -> gecode_constraint_path_243(Y0,Y1,Y2,Y3,Y4,Y5,Y6,Y7)
                                     ;  throw(error(type_error('IntVar'(X7)),gecode_argument_error(path(X0,X1,X2,X3,X4,X5,X6,X7),arg=8))))
                                 ;  throw(error(type_error('IntVarArgs'(X6)),gecode_argument_error(path(X0,X1,X2,X3,X4,X5,X6,X7),arg=7))))
                             ;  throw(error(type_error('IntVar'(X5)),gecode_argument_error(path(X0,X1,X2,X3,X4,X5,X6,X7),arg=6))))
                         ;  throw(error(type_error('IntVar'(X4)),gecode_argument_error(path(X0,X1,X2,X3,X4,X5,X6,X7),arg=5))))
                     ;  throw(error(type_error('IntVarArgs'(X3)),gecode_argument_error(path(X0,X1,X2,X3,X4,X5,X6,X7),arg=4))))
                 ;  throw(error(type_error(int(X2)),gecode_argument_error(path(X0,X1,X2,X3,X4,X5,X6,X7),arg=3))))
             ;  throw(error(type_error('IntArgs'(X1)),gecode_argument_error(path(X0,X1,X2,X3,X4,X5,X6,X7),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(path(X0,X1,X2,X3,X4,X5,X6,X7),arg=1)))).

path(X0,X1,X2,X3) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_IntVarArgs(X1,Y1)
             -> (is_IntVar(X2,Y2)
                 -> (is_IntVar(X3,Y3)
                     -> gecode_constraint_path_245(Y0,Y1,Y2,Y3)
                     ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(path(X0,X1,X2,X3),arg=4))))
                 ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(path(X0,X1,X2,X3),arg=3))))
             ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(path(X0,X1,X2,X3),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(path(X0,X1,X2,X3),arg=1)))).

path(X0,X1,X2,X3,X4) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_int(X1,Y1)
             -> (is_IntVarArgs(X2,Y2)
                 -> (is_IntVar(X3,Y3)
                     -> (is_IntVar(X4,Y4)
                         -> gecode_constraint_path_246(Y0,Y1,Y2,Y3,Y4)
                         ;  throw(error(type_error('IntVar'(X4)),gecode_argument_error(path(X0,X1,X2,X3,X4),arg=5))))
                     ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(path(X0,X1,X2,X3,X4),arg=4))))
                 ;  throw(error(type_error('IntVarArgs'(X2)),gecode_argument_error(path(X0,X1,X2,X3,X4),arg=3))))
             ;  throw(error(type_error(int(X1)),gecode_argument_error(path(X0,X1,X2,X3,X4),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(path(X0,X1,X2,X3,X4),arg=1)))).

pow(X0,X1,X2,X3) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_FloatVar(X1,Y1)
             -> (is_int(X2,Y2)
                 -> (is_FloatVar(X3,Y3)
                     -> gecode_constraint_pow_247(Y0,Y1,Y2,Y3)
                     ;  throw(error(type_error('FloatVar'(X3)),gecode_argument_error(pow(X0,X1,X2,X3),arg=4))))
                 ;  throw(error(type_error(int(X2)),gecode_argument_error(pow(X0,X1,X2,X3),arg=3))))
             ;  (is_IntVar(X1,Y1)
                 -> (is_int(X2,Y2)
                     -> (is_IntVar(X3,Y3)
                         -> gecode_constraint_pow_248(Y0,Y1,Y2,Y3)
                         ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(pow(X0,X1,X2,X3),arg=4))))
                     ;  throw(error(type_error(int(X2)),gecode_argument_error(pow(X0,X1,X2,X3),arg=3))))
                 ;  throw(error(type_error('IntVar'(X1)),gecode_argument_error(pow(X0,X1,X2,X3),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(pow(X0,X1,X2,X3),arg=1)))).

precede(X0,X1,X2) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_IntVarArgs(X1,Y1)
             -> (is_IntArgs(X2,Y2)
                 -> gecode_constraint_precede_249(Y0,Y1,Y2)
                 ;  throw(error(type_error('IntArgs'(X2)),gecode_argument_error(precede(X0,X1,X2),arg=3))))
             ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(precede(X0,X1,X2),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(precede(X0,X1,X2),arg=1)))).

precede(X0,X1,X2,X3) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_IntVarArgs(X1,Y1)
             -> (is_int(X2,Y2)
                 -> (is_int(X3,Y3)
                     -> gecode_constraint_precede_250(Y0,Y1,Y2,Y3)
                     ;  throw(error(type_error(int(X3)),gecode_argument_error(precede(X0,X1,X2,X3),arg=4))))
                 ;  throw(error(type_error(int(X2)),gecode_argument_error(precede(X0,X1,X2,X3),arg=3))))
             ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(precede(X0,X1,X2,X3),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(precede(X0,X1,X2,X3),arg=1)))).

relax(X0,X1,X2,X3,X4) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVarArgs(X1,Y1)
             -> (is_BoolVarArgs(X2,Y2)
                 -> (is_Rnd(X3,Y3)
                     -> (is_double(X4,Y4)
                         -> gecode_constraint_relax_251(Y0,Y1,Y2,Y3,Y4)
                         ;  throw(error(type_error(double(X4)),gecode_argument_error(relax(X0,X1,X2,X3,X4),arg=5))))
                     ;  throw(error(type_error('Rnd'(X3)),gecode_argument_error(relax(X0,X1,X2,X3,X4),arg=4))))
                 ;  throw(error(type_error('BoolVarArgs'(X2)),gecode_argument_error(relax(X0,X1,X2,X3,X4),arg=3))))
             ;  (is_FloatVarArgs(X1,Y1)
                 -> (is_FloatVarArgs(X2,Y2)
                     -> (is_Rnd(X3,Y3)
                         -> (is_double(X4,Y4)
                             -> gecode_constraint_relax_252(Y0,Y1,Y2,Y3,Y4)
                             ;  throw(error(type_error(double(X4)),gecode_argument_error(relax(X0,X1,X2,X3,X4),arg=5))))
                         ;  throw(error(type_error('Rnd'(X3)),gecode_argument_error(relax(X0,X1,X2,X3,X4),arg=4))))
                     ;  throw(error(type_error('FloatVarArgs'(X2)),gecode_argument_error(relax(X0,X1,X2,X3,X4),arg=3))))
                 ;  (is_IntVarArgs(X1,Y1)
                     -> (is_IntVarArgs(X2,Y2)
                         -> (is_Rnd(X3,Y3)
                             -> (is_double(X4,Y4)
                                 -> gecode_constraint_relax_253(Y0,Y1,Y2,Y3,Y4)
                                 ;  throw(error(type_error(double(X4)),gecode_argument_error(relax(X0,X1,X2,X3,X4),arg=5))))
                             ;  throw(error(type_error('Rnd'(X3)),gecode_argument_error(relax(X0,X1,X2,X3,X4),arg=4))))
                         ;  throw(error(type_error('IntVarArgs'(X2)),gecode_argument_error(relax(X0,X1,X2,X3,X4),arg=3))))
                     ;  (is_SetVarArgs(X1,Y1)
                         -> (is_SetVarArgs(X2,Y2)
                             -> (is_Rnd(X3,Y3)
                                 -> (is_double(X4,Y4)
                                     -> gecode_constraint_relax_254(Y0,Y1,Y2,Y3,Y4)
                                     ;  throw(error(type_error(double(X4)),gecode_argument_error(relax(X0,X1,X2,X3,X4),arg=5))))
                                 ;  throw(error(type_error('Rnd'(X3)),gecode_argument_error(relax(X0,X1,X2,X3,X4),arg=4))))
                             ;  throw(error(type_error('SetVarArgs'(X2)),gecode_argument_error(relax(X0,X1,X2,X3,X4),arg=3))))
                         ;  throw(error(type_error('SetVarArgs'(X1)),gecode_argument_error(relax(X0,X1,X2,X3,X4),arg=2)))))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(relax(X0,X1,X2,X3,X4),arg=1)))).

rel(X0,X1,X2,X3) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolOpType(X1,Y1)
             -> (is_BoolVarArgs(X2,Y2)
                 -> (is_BoolVar(X3,Y3)
                     -> gecode_constraint_rel_255(Y0,Y1,Y2,Y3)
                     ;  (is_int(X3,Y3)
                         -> gecode_constraint_rel_256(Y0,Y1,Y2,Y3)
                         ;  throw(error(type_error(int(X3)),gecode_argument_error(rel(X0,X1,X2,X3),arg=4)))))
                 ;  throw(error(type_error('BoolVarArgs'(X2)),gecode_argument_error(rel(X0,X1,X2,X3),arg=3))))
             ;  (is_BoolVar(X1,Y1)
                 -> (is_IntRelType(X2,Y2)
                     -> (is_BoolVar(X3,Y3)
                         -> gecode_constraint_rel_259(Y0,Y1,Y2,Y3)
                         ;  (is_int(X3,Y3)
                             -> gecode_constraint_rel_261(Y0,Y1,Y2,Y3)
                             ;  throw(error(type_error(int(X3)),gecode_argument_error(rel(X0,X1,X2,X3),arg=4)))))
                     ;  throw(error(type_error('IntRelType'(X2)),gecode_argument_error(rel(X0,X1,X2,X3),arg=3))))
                 ;  (is_BoolVarArgs(X1,Y1)
                     -> (is_IntRelType(X2,Y2)
                         -> (is_BoolVar(X3,Y3)
                             -> gecode_constraint_rel_263(Y0,Y1,Y2,Y3)
                             ;  (is_BoolVarArgs(X3,Y3)
                                 -> gecode_constraint_rel_264(Y0,Y1,Y2,Y3)
                                 ;  (is_IntArgs(X3,Y3)
                                     -> gecode_constraint_rel_265(Y0,Y1,Y2,Y3)
                                     ;  (is_int(X3,Y3)
                                         -> gecode_constraint_rel_266(Y0,Y1,Y2,Y3)
                                         ;  throw(error(type_error(int(X3)),gecode_argument_error(rel(X0,X1,X2,X3),arg=4)))))))
                         ;  throw(error(type_error('IntRelType'(X2)),gecode_argument_error(rel(X0,X1,X2,X3),arg=3))))
                     ;  (is_FloatVarArgs(X1,Y1)
                         -> (is_FloatRelType(X2,Y2)
                             -> (is_FloatVal(X3,Y3)
                                 -> gecode_constraint_rel_268(Y0,Y1,Y2,Y3)
                                 ;  (is_FloatVar(X3,Y3)
                                     -> gecode_constraint_rel_269(Y0,Y1,Y2,Y3)
                                     ;  throw(error(type_error('FloatVar'(X3)),gecode_argument_error(rel(X0,X1,X2,X3),arg=4)))))
                             ;  throw(error(type_error('FloatRelType'(X2)),gecode_argument_error(rel(X0,X1,X2,X3),arg=3))))
                         ;  (is_IntArgs(X1,Y1)
                             -> (is_IntRelType(X2,Y2)
                                 -> (is_BoolVarArgs(X3,Y3)
                                     -> gecode_constraint_rel_270(Y0,Y1,Y2,Y3)
                                     ;  (is_IntVarArgs(X3,Y3)
                                         -> gecode_constraint_rel_271(Y0,Y1,Y2,Y3)
                                         ;  throw(error(type_error('IntVarArgs'(X3)),gecode_argument_error(rel(X0,X1,X2,X3),arg=4)))))
                                 ;  throw(error(type_error('IntRelType'(X2)),gecode_argument_error(rel(X0,X1,X2,X3),arg=3))))
                             ;  (is_IntVarArgs(X1,Y1)
                                 -> (is_IntRelType(X2,Y2)
                                     -> (is_IntArgs(X3,Y3)
                                         -> gecode_constraint_rel_272(Y0,Y1,Y2,Y3)
                                         ;  (is_IntVarArgs(X3,Y3)
                                             -> gecode_constraint_rel_273(Y0,Y1,Y2,Y3)
                                             ;  (is_int(X3,Y3)
                                                 -> gecode_constraint_rel_274(Y0,Y1,Y2,Y3)
                                                 ;  (is_IntVar(X3,Y3)
                                                     -> gecode_constraint_rel_276(Y0,Y1,Y2,Y3)
                                                     ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(rel(X0,X1,X2,X3),arg=4)))))))
                                     ;  throw(error(type_error('IntRelType'(X2)),gecode_argument_error(rel(X0,X1,X2,X3),arg=3))))
                                 ;  (is_FloatVar(X1,Y1)
                                     -> (is_FloatRelType(X2,Y2)
                                         -> (is_FloatVal(X3,Y3)
                                             -> gecode_constraint_rel_277(Y0,Y1,Y2,Y3)
                                             ;  (is_FloatVar(X3,Y3)
                                                 -> gecode_constraint_rel_279(Y0,Y1,Y2,Y3)
                                                 ;  throw(error(type_error('FloatVar'(X3)),gecode_argument_error(rel(X0,X1,X2,X3),arg=4)))))
                                         ;  throw(error(type_error('FloatRelType'(X2)),gecode_argument_error(rel(X0,X1,X2,X3),arg=3))))
                                     ;  (is_IntVar(X1,Y1)
                                         -> (is_IntRelType(X2,Y2)
                                             -> (is_int(X3,Y3)
                                                 -> gecode_constraint_rel_281(Y0,Y1,Y2,Y3)
                                                 ;  (is_IntVar(X3,Y3)
                                                     -> gecode_constraint_rel_283(Y0,Y1,Y2,Y3)
                                                     ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(rel(X0,X1,X2,X3),arg=4)))))
                                             ;  (is_SetRelType(X2,Y2)
                                                 -> (is_SetVar(X3,Y3)
                                                     -> gecode_constraint_rel_285(Y0,Y1,Y2,Y3)
                                                     ;  throw(error(type_error('SetVar'(X3)),gecode_argument_error(rel(X0,X1,X2,X3),arg=4))))
                                                 ;  throw(error(type_error('SetRelType'(X2)),gecode_argument_error(rel(X0,X1,X2,X3),arg=3)))))
                                         ;  (is_SetVar(X1,Y1)
                                             -> (is_IntRelType(X2,Y2)
                                                 -> (is_IntVar(X3,Y3)
                                                     -> gecode_constraint_rel_287(Y0,Y1,Y2,Y3)
                                                     ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(rel(X0,X1,X2,X3),arg=4))))
                                                 ;  (is_SetRelType(X2,Y2)
                                                     -> (is_IntVar(X3,Y3)
                                                         -> gecode_constraint_rel_289(Y0,Y1,Y2,Y3)
                                                         ;  (is_SetVar(X3,Y3)
                                                             -> gecode_constraint_rel_291(Y0,Y1,Y2,Y3)
                                                             ;  throw(error(type_error('SetVar'(X3)),gecode_argument_error(rel(X0,X1,X2,X3),arg=4)))))
                                                     ;  throw(error(type_error('SetRelType'(X2)),gecode_argument_error(rel(X0,X1,X2,X3),arg=3)))))
                                             ;  throw(error(type_error('SetVar'(X1)),gecode_argument_error(rel(X0,X1,X2,X3),arg=2))))))))))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(rel(X0,X1,X2,X3),arg=1)))).

rel(X0,X1,X2,X3,X4) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVar(X1,Y1)
             -> (is_BoolOpType(X2,Y2)
                 -> (is_BoolVar(X3,Y3)
                     -> (is_BoolVar(X4,Y4)
                         -> gecode_constraint_rel_257(Y0,Y1,Y2,Y3,Y4)
                         ;  (is_int(X4,Y4)
                             -> gecode_constraint_rel_258(Y0,Y1,Y2,Y3,Y4)
                             ;  throw(error(type_error(int(X4)),gecode_argument_error(rel(X0,X1,X2,X3,X4),arg=5)))))
                     ;  throw(error(type_error('BoolVar'(X3)),gecode_argument_error(rel(X0,X1,X2,X3,X4),arg=4))))
                 ;  (is_IntRelType(X2,Y2)
                     -> (is_BoolVar(X3,Y3)
                         -> (is_Reify(X4,Y4)
                             -> gecode_constraint_rel_260(Y0,Y1,Y2,Y3,Y4)
                             ;  throw(error(type_error('Reify'(X4)),gecode_argument_error(rel(X0,X1,X2,X3,X4),arg=5))))
                         ;  (is_int(X3,Y3)
                             -> (is_Reify(X4,Y4)
                                 -> gecode_constraint_rel_262(Y0,Y1,Y2,Y3,Y4)
                                 ;  throw(error(type_error('Reify'(X4)),gecode_argument_error(rel(X0,X1,X2,X3,X4),arg=5))))
                             ;  throw(error(type_error(int(X3)),gecode_argument_error(rel(X0,X1,X2,X3,X4),arg=4)))))
                     ;  throw(error(type_error('IntRelType'(X2)),gecode_argument_error(rel(X0,X1,X2,X3,X4),arg=3)))))
             ;  (is_FloatVar(X1,Y1)
                 -> (is_FloatRelType(X2,Y2)
                     -> (is_FloatVal(X3,Y3)
                         -> (is_Reify(X4,Y4)
                             -> gecode_constraint_rel_278(Y0,Y1,Y2,Y3,Y4)
                             ;  throw(error(type_error('Reify'(X4)),gecode_argument_error(rel(X0,X1,X2,X3,X4),arg=5))))
                         ;  (is_FloatVar(X3,Y3)
                             -> (is_Reify(X4,Y4)
                                 -> gecode_constraint_rel_280(Y0,Y1,Y2,Y3,Y4)
                                 ;  throw(error(type_error('Reify'(X4)),gecode_argument_error(rel(X0,X1,X2,X3,X4),arg=5))))
                             ;  throw(error(type_error('FloatVar'(X3)),gecode_argument_error(rel(X0,X1,X2,X3,X4),arg=4)))))
                     ;  throw(error(type_error('FloatRelType'(X2)),gecode_argument_error(rel(X0,X1,X2,X3,X4),arg=3))))
                 ;  (is_IntVar(X1,Y1)
                     -> (is_IntRelType(X2,Y2)
                         -> (is_int(X3,Y3)
                             -> (is_Reify(X4,Y4)
                                 -> gecode_constraint_rel_282(Y0,Y1,Y2,Y3,Y4)
                                 ;  throw(error(type_error('Reify'(X4)),gecode_argument_error(rel(X0,X1,X2,X3,X4),arg=5))))
                             ;  (is_IntVar(X3,Y3)
                                 -> (is_Reify(X4,Y4)
                                     -> gecode_constraint_rel_284(Y0,Y1,Y2,Y3,Y4)
                                     ;  throw(error(type_error('Reify'(X4)),gecode_argument_error(rel(X0,X1,X2,X3,X4),arg=5))))
                                 ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(rel(X0,X1,X2,X3,X4),arg=4)))))
                         ;  (is_SetRelType(X2,Y2)
                             -> (is_SetVar(X3,Y3)
                                 -> (is_Reify(X4,Y4)
                                     -> gecode_constraint_rel_286(Y0,Y1,Y2,Y3,Y4)
                                     ;  throw(error(type_error('Reify'(X4)),gecode_argument_error(rel(X0,X1,X2,X3,X4),arg=5))))
                                 ;  throw(error(type_error('SetVar'(X3)),gecode_argument_error(rel(X0,X1,X2,X3,X4),arg=4))))
                             ;  throw(error(type_error('SetRelType'(X2)),gecode_argument_error(rel(X0,X1,X2,X3,X4),arg=3)))))
                     ;  (is_SetVar(X1,Y1)
                         -> (is_IntRelType(X2,Y2)
                             -> (is_IntVar(X3,Y3)
                                 -> (is_Reify(X4,Y4)
                                     -> gecode_constraint_rel_288(Y0,Y1,Y2,Y3,Y4)
                                     ;  throw(error(type_error('Reify'(X4)),gecode_argument_error(rel(X0,X1,X2,X3,X4),arg=5))))
                                 ;  throw(error(type_error('IntVar'(X3)),gecode_argument_error(rel(X0,X1,X2,X3,X4),arg=4))))
                             ;  (is_SetRelType(X2,Y2)
                                 -> (is_IntVar(X3,Y3)
                                     -> (is_Reify(X4,Y4)
                                         -> gecode_constraint_rel_290(Y0,Y1,Y2,Y3,Y4)
                                         ;  throw(error(type_error('Reify'(X4)),gecode_argument_error(rel(X0,X1,X2,X3,X4),arg=5))))
                                     ;  (is_SetVar(X3,Y3)
                                         -> (is_Reify(X4,Y4)
                                             -> gecode_constraint_rel_292(Y0,Y1,Y2,Y3,Y4)
                                             ;  throw(error(type_error('Reify'(X4)),gecode_argument_error(rel(X0,X1,X2,X3,X4),arg=5))))
                                         ;  throw(error(type_error('SetVar'(X3)),gecode_argument_error(rel(X0,X1,X2,X3,X4),arg=4)))))
                                 ;  throw(error(type_error('SetRelType'(X2)),gecode_argument_error(rel(X0,X1,X2,X3,X4),arg=3)))))
                         ;  throw(error(type_error('SetVar'(X1)),gecode_argument_error(rel(X0,X1,X2,X3,X4),arg=2)))))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(rel(X0,X1,X2,X3,X4),arg=1)))).

rel(X0,X1,X2) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVarArgs(X1,Y1)
             -> (is_IntRelType(X2,Y2)
                 -> gecode_constraint_rel_267(Y0,Y1,Y2)
                 ;  throw(error(type_error('IntRelType'(X2)),gecode_argument_error(rel(X0,X1,X2),arg=3))))
             ;  (is_IntVarArgs(X1,Y1)
                 -> (is_IntRelType(X2,Y2)
                     -> gecode_constraint_rel_275(Y0,Y1,Y2)
                     ;  throw(error(type_error('IntRelType'(X2)),gecode_argument_error(rel(X0,X1,X2),arg=3))))
                 ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(rel(X0,X1,X2),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(rel(X0,X1,X2),arg=1)))).

sequence(X0,X1,X2,X3,X4,X5) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVarArgs(X1,Y1)
             -> (is_IntSet(X2,Y2)
                 -> (is_int(X3,Y3)
                     -> (is_int(X4,Y4)
                         -> (is_int(X5,Y5)
                             -> gecode_constraint_sequence_293(Y0,Y1,Y2,Y3,Y4,Y5)
                             ;  throw(error(type_error(int(X5)),gecode_argument_error(sequence(X0,X1,X2,X3,X4,X5),arg=6))))
                         ;  throw(error(type_error(int(X4)),gecode_argument_error(sequence(X0,X1,X2,X3,X4,X5),arg=5))))
                     ;  throw(error(type_error(int(X3)),gecode_argument_error(sequence(X0,X1,X2,X3,X4,X5),arg=4))))
                 ;  throw(error(type_error('IntSet'(X2)),gecode_argument_error(sequence(X0,X1,X2,X3,X4,X5),arg=3))))
             ;  (is_IntVarArgs(X1,Y1)
                 -> (is_IntSet(X2,Y2)
                     -> (is_int(X3,Y3)
                         -> (is_int(X4,Y4)
                             -> (is_int(X5,Y5)
                                 -> gecode_constraint_sequence_294(Y0,Y1,Y2,Y3,Y4,Y5)
                                 ;  throw(error(type_error(int(X5)),gecode_argument_error(sequence(X0,X1,X2,X3,X4,X5),arg=6))))
                             ;  throw(error(type_error(int(X4)),gecode_argument_error(sequence(X0,X1,X2,X3,X4,X5),arg=5))))
                         ;  throw(error(type_error(int(X3)),gecode_argument_error(sequence(X0,X1,X2,X3,X4,X5),arg=4))))
                     ;  throw(error(type_error('IntSet'(X2)),gecode_argument_error(sequence(X0,X1,X2,X3,X4,X5),arg=3))))
                 ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(sequence(X0,X1,X2,X3,X4,X5),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(sequence(X0,X1,X2,X3,X4,X5),arg=1)))).

sorted(X0,X1,X2,X3) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_IntVarArgs(X1,Y1)
             -> (is_IntVarArgs(X2,Y2)
                 -> (is_IntVarArgs(X3,Y3)
                     -> gecode_constraint_sorted_295(Y0,Y1,Y2,Y3)
                     ;  throw(error(type_error('IntVarArgs'(X3)),gecode_argument_error(sorted(X0,X1,X2,X3),arg=4))))
                 ;  throw(error(type_error('IntVarArgs'(X2)),gecode_argument_error(sorted(X0,X1,X2,X3),arg=3))))
             ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(sorted(X0,X1,X2,X3),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(sorted(X0,X1,X2,X3),arg=1)))).

sorted(X0,X1,X2) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_IntVarArgs(X1,Y1)
             -> (is_IntVarArgs(X2,Y2)
                 -> gecode_constraint_sorted_296(Y0,Y1,Y2)
                 ;  throw(error(type_error('IntVarArgs'(X2)),gecode_argument_error(sorted(X0,X1,X2),arg=3))))
             ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(sorted(X0,X1,X2),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(sorted(X0,X1,X2),arg=1)))).

sqr(X0,X1,X2) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_FloatVar(X1,Y1)
             -> (is_FloatVar(X2,Y2)
                 -> gecode_constraint_sqr_297(Y0,Y1,Y2)
                 ;  throw(error(type_error('FloatVar'(X2)),gecode_argument_error(sqr(X0,X1,X2),arg=3))))
             ;  (is_IntVar(X1,Y1)
                 -> (is_IntVar(X2,Y2)
                     -> gecode_constraint_sqr_298(Y0,Y1,Y2)
                     ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(sqr(X0,X1,X2),arg=3))))
                 ;  throw(error(type_error('IntVar'(X1)),gecode_argument_error(sqr(X0,X1,X2),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(sqr(X0,X1,X2),arg=1)))).

sqrt(X0,X1,X2) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_FloatVar(X1,Y1)
             -> (is_FloatVar(X2,Y2)
                 -> gecode_constraint_sqrt_299(Y0,Y1,Y2)
                 ;  throw(error(type_error('FloatVar'(X2)),gecode_argument_error(sqrt(X0,X1,X2),arg=3))))
             ;  (is_IntVar(X1,Y1)
                 -> (is_IntVar(X2,Y2)
                     -> gecode_constraint_sqrt_300(Y0,Y1,Y2)
                     ;  throw(error(type_error('IntVar'(X2)),gecode_argument_error(sqrt(X0,X1,X2),arg=3))))
                 ;  throw(error(type_error('IntVar'(X1)),gecode_argument_error(sqrt(X0,X1,X2),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(sqrt(X0,X1,X2),arg=1)))).

unary(X0,X1,X2,X3) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_IntVarArgs(X1,Y1)
             -> (is_IntArgs(X2,Y2)
                 -> (is_BoolVarArgs(X3,Y3)
                     -> gecode_constraint_unary_301(Y0,Y1,Y2,Y3)
                     ;  throw(error(type_error('BoolVarArgs'(X3)),gecode_argument_error(unary(X0,X1,X2,X3),arg=4))))
                 ;  (is_IntVarArgs(X2,Y2)
                     -> (is_IntVarArgs(X3,Y3)
                         -> gecode_constraint_unary_304(Y0,Y1,Y2,Y3)
                         ;  throw(error(type_error('IntVarArgs'(X3)),gecode_argument_error(unary(X0,X1,X2,X3),arg=4))))
                     ;  throw(error(type_error('IntVarArgs'(X2)),gecode_argument_error(unary(X0,X1,X2,X3),arg=3)))))
             ;  (is_TaskTypeArgs(X1,Y1)
                 -> (is_IntVarArgs(X2,Y2)
                     -> (is_IntArgs(X3,Y3)
                         -> gecode_constraint_unary_306(Y0,Y1,Y2,Y3)
                         ;  throw(error(type_error('IntArgs'(X3)),gecode_argument_error(unary(X0,X1,X2,X3),arg=4))))
                     ;  throw(error(type_error('IntVarArgs'(X2)),gecode_argument_error(unary(X0,X1,X2,X3),arg=3))))
                 ;  throw(error(type_error('TaskTypeArgs'(X1)),gecode_argument_error(unary(X0,X1,X2,X3),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(unary(X0,X1,X2,X3),arg=1)))).

unary(X0,X1,X2) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_IntVarArgs(X1,Y1)
             -> (is_IntArgs(X2,Y2)
                 -> gecode_constraint_unary_302(Y0,Y1,Y2)
                 ;  throw(error(type_error('IntArgs'(X2)),gecode_argument_error(unary(X0,X1,X2),arg=3))))
             ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(unary(X0,X1,X2),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(unary(X0,X1,X2),arg=1)))).

unary(X0,X1,X2,X3,X4) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_IntVarArgs(X1,Y1)
             -> (is_IntVarArgs(X2,Y2)
                 -> (is_IntVarArgs(X3,Y3)
                     -> (is_BoolVarArgs(X4,Y4)
                         -> gecode_constraint_unary_303(Y0,Y1,Y2,Y3,Y4)
                         ;  throw(error(type_error('BoolVarArgs'(X4)),gecode_argument_error(unary(X0,X1,X2,X3,X4),arg=5))))
                     ;  throw(error(type_error('IntVarArgs'(X3)),gecode_argument_error(unary(X0,X1,X2,X3,X4),arg=4))))
                 ;  throw(error(type_error('IntVarArgs'(X2)),gecode_argument_error(unary(X0,X1,X2,X3,X4),arg=3))))
             ;  (is_TaskTypeArgs(X1,Y1)
                 -> (is_IntVarArgs(X2,Y2)
                     -> (is_IntArgs(X3,Y3)
                         -> (is_BoolVarArgs(X4,Y4)
                             -> gecode_constraint_unary_305(Y0,Y1,Y2,Y3,Y4)
                             ;  throw(error(type_error('BoolVarArgs'(X4)),gecode_argument_error(unary(X0,X1,X2,X3,X4),arg=5))))
                         ;  throw(error(type_error('IntArgs'(X3)),gecode_argument_error(unary(X0,X1,X2,X3,X4),arg=4))))
                     ;  throw(error(type_error('IntVarArgs'(X2)),gecode_argument_error(unary(X0,X1,X2,X3,X4),arg=3))))
                 ;  throw(error(type_error('TaskTypeArgs'(X1)),gecode_argument_error(unary(X0,X1,X2,X3,X4),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(unary(X0,X1,X2,X3,X4),arg=1)))).

unshare(X0,X1) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVarArgs(X1,Y1)
             -> gecode_constraint_unshare_307(Y0,Y1)
             ;  (is_IntVarArgs(X1,Y1)
                 -> gecode_constraint_unshare_308(Y0,Y1)
                 ;  throw(error(type_error('IntVarArgs'(X1)),gecode_argument_error(unshare(X0,X1),arg=2)))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(unshare(X0,X1),arg=1)))).

wait(X0,X1,X2) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVar(X1,Y1)
             -> (is_std_function(X2,Y2)
                 -> gecode_constraint_wait_309(Y0,Y1,Y2)
                 ;  throw(error(type_error('std::function<void(Space&home)>'(X2)),gecode_argument_error(wait(X0,X1,X2),arg=3))))
             ;  (is_BoolVarArgs(X1,Y1)
                 -> (is_std_function(X2,Y2)
                     -> gecode_constraint_wait_310(Y0,Y1,Y2)
                     ;  throw(error(type_error('std::function<void(Space&home)>'(X2)),gecode_argument_error(wait(X0,X1,X2),arg=3))))
                 ;  (is_FloatVarArgs(X1,Y1)
                     -> (is_std_function(X2,Y2)
                         -> gecode_constraint_wait_311(Y0,Y1,Y2)
                         ;  throw(error(type_error('std::function<void(Space&home)>'(X2)),gecode_argument_error(wait(X0,X1,X2),arg=3))))
                     ;  (is_IntVarArgs(X1,Y1)
                         -> (is_std_function(X2,Y2)
                             -> gecode_constraint_wait_312(Y0,Y1,Y2)
                             ;  throw(error(type_error('std::function<void(Space&home)>'(X2)),gecode_argument_error(wait(X0,X1,X2),arg=3))))
                         ;  (is_FloatVar(X1,Y1)
                             -> (is_std_function(X2,Y2)
                                 -> gecode_constraint_wait_313(Y0,Y1,Y2)
                                 ;  throw(error(type_error('std::function<void(Space&home)>'(X2)),gecode_argument_error(wait(X0,X1,X2),arg=3))))
                             ;  (is_IntVar(X1,Y1)
                                 -> (is_std_function(X2,Y2)
                                     -> gecode_constraint_wait_314(Y0,Y1,Y2)
                                     ;  throw(error(type_error('std::function<void(Space&home)>'(X2)),gecode_argument_error(wait(X0,X1,X2),arg=3))))
                                 ;  throw(error(type_error('IntVar'(X1)),gecode_argument_error(wait(X0,X1,X2),arg=2)))))))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(wait(X0,X1,X2),arg=1)))).

when(X0,X1,X2) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVar(X1,Y1)
             -> (is_std_function(X2,Y2)
                 -> gecode_constraint_when_315(Y0,Y1,Y2)
                 ;  throw(error(type_error('std::function<void(Space&home)>'(X2)),gecode_argument_error(when(X0,X1,X2),arg=3))))
             ;  throw(error(type_error('BoolVar'(X1)),gecode_argument_error(when(X0,X1,X2),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(when(X0,X1,X2),arg=1)))).

when(X0,X1,X2,X3) :-
        (is_Space_or_Clause(X0,Y0)
         -> (is_BoolVar(X1,Y1)
             -> (is_std_function(X2,Y2)
                 -> (is_std_function(X3,Y3)
                     -> gecode_constraint_when_316(Y0,Y1,Y2,Y3)
                     ;  throw(error(type_error('std::function<void(Space&home)>'(X3)),gecode_argument_error(when(X0,X1,X2,X3),arg=4))))
                 ;  throw(error(type_error('std::function<void(Space&home)>'(X2)),gecode_argument_error(when(X0,X1,X2,X3),arg=3))))
             ;  throw(error(type_error('BoolVar'(X1)),gecode_argument_error(when(X0,X1,X2,X3),arg=2))))
         ;  throw(error(type_error('Space'(X0)),gecode_argument_error(when(X0,X1,X2,X3),arg=1)))).

