# Copyright © 2025 Haocheng Wen, Faxuan Luo
# SPDX-License-Identifier: MIT

import jax.numpy as jnp
from jax import jit,vmap,pmap
import janc.aux_func as aux_func
from janc.flux import weno5
import janc.thermo as thermo
import janc.chemical as chemical
import janc.boundary as boundary
from functools import partial

from janc.config import template_node_num
import janc.jaxamr as amr

def set_solver(thermo_set,boundary_set,source_set,solver_mode):
    thermo.set_thermo(thermo_set)
    boundary.set_boundary(boundary_set)
    aux_func.set_source_terms(source_set)
    if thermo.thermo_settings['is_detailed_chemistry']:
        chem_solver_type = 'implicit'
    else:
        chem_solver_type = 'explicit'
    
    def CFL(field,dx,cfl=0.20):
        U, aux = field[0:-2],field[-2:]
        _,u,_,_,_,a = aux_func.U_to_prim(U,aux)
        c = jnp.max(jnp.abs(u) + a)
        dt = cfl/c*dx
        return dt
    
    if solver_mode == 'amr':
        
        @partial(vmap,in_axes=(0, 0, None,None))
        def rhs(U, aux, dx, dy):
            aux = aux_func.update_aux(U, aux)
            physical_rhs = weno5(U,aux,dx,dy) + aux_func.source_terms(U[:,3:-3,3:-3], aux[:,3:-3,3:-3])
            return jnp.pad(physical_rhs,pad_width=((0,0),(3,3),(3,3)))
    
    else:
        if solver_mode == 'parallel':
            def rhs(U,aux,dx,dy):
                aux = aux_func.update_aux(U, aux)
                U_with_ghost,aux_with_ghost = boundary.parallel_boundary_conditions(U,aux)
                physical_rhs = weno5(U_with_ghost,aux_with_ghost,dx,dy) + aux_func.source_terms(U, aux)
                return physical_rhs
        else:
            def rhs(U,aux,dx,dy):
                aux = aux_func.update_aux(U, aux)
                U_with_ghost,aux_with_ghost = boundary.boundary_conditions(U,aux)
                physical_rhs = weno5(U_with_ghost,aux_with_ghost,dx,dy) + aux_func.source_terms(U, aux)
                return physical_rhs
    
    if solver_mode == 'amr':
        def advance_flux(level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info):

            num = template_node_num

            ghost_blk_data = amr.get_ghost_block_data(ref_blk_data, ref_blk_info)
            U,aux = ghost_blk_data[:,0:-2],ghost_blk_data[:,-2:]
            U1 = U + dt * rhs(U, aux, dx, dy)
            blk_data1 = jnp.concatenate([U1,aux],axis=1)
            blk_data1 = amr.update_external_boundary(level, blk_data, blk_data1[..., num:-num, num:-num], ref_blk_info)


            ghost_blk_data1 = amr.get_ghost_block_data(blk_data1, ref_blk_info)
            U1 = ghost_blk_data1[:,0:-2]
            U2 = 3/4*U + 1/4*(U1 + dt * rhs(U1, aux, dx, dy))
            blk_data2 = jnp.concatenate([U2,aux],axis=1)
            blk_data2 = amr.update_external_boundary(level, blk_data, blk_data2[..., num:-num, num:-num], ref_blk_info)

            ghost_blk_data2 = amr.get_ghost_block_data(blk_data2, ref_blk_info)
            U2 = ghost_blk_data2[:,0:-2]
            U3 = 1/3*U + 2/3*(U2 + dt * rhs(U2, aux, dx, dy))
            blk_data3 = jnp.concatenate([U3,aux],axis=1)
            blk_data3 = amr.update_external_boundary(level, blk_data, blk_data3[..., num:-num, num:-num], ref_blk_info)
            
            return blk_data3
    
    else:
        def advance_flux(field,dx,dy,dt):
            
            U, aux = field[0:-2],field[-2:]
            U1 = U + dt * rhs(U,aux,dx,dy)
            U2 = 3/4*U + 1/4 * (U1 + dt * rhs(U1,aux,dx,dy))
            U3 = 1/3*U + 2/3 * (U2 + dt * rhs(U2,aux,dx,dy))
            field = jnp.concatenate([U3,aux],axis=0)
            
            return field
    
    def advance_source_term(field,dt):
        U, aux = field[0:-2],field[-2:]
        aux = aux_func.update_aux(U, aux)
        _,T = aux_func.aux_to_thermo(U,aux)
        rho = U[0:1]
        Y = U[4:]/rho
        drhoY = chemical.solve_implicit_rate(T,rho,Y,dt)

        p1 = U[0:4,:,:]
        p2 = U[4:,:,:] + drhoY
        U_new = jnp.concatenate([p1,p2],axis=0)
        return jnp.concatenate([U_new,aux],axis=0)

    
    if chem_solver_type == 'implicit':
        if solver_mode == 'amr':
            @partial(jit,static_argnames='level')
            def advance_one_step(level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info):
                field_adv = advance_flux(level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info)
                field = vmap(advance_source_term,in_axes=(0, None))(field_adv,dt)
                return field
        
        else:
            if solver_mode == 'parallel':
                @partial(pmap,axis_name='x')    
                @jit    
                def advance_one_step(field,dx,dy,dt):
                    field_adv = advance_flux(field,dx,dy,dt)
                    field = advance_source_term(field_adv,dt)
                    return field
            else:
                @jit    
                def advance_one_step(field,dx,dy,dt):
                    field_adv = advance_flux(field,dx,dy,dt)
                    field = advance_source_term(field_adv,dt)
                    return field
    else:
        if solver_mode == 'amr':
            @partial(jit,static_argnames='level')
            def advance_one_step(level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info):
                field = advance_flux(level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info)
                return field
        
        else:
            if solver_mode == 'parallel':
                @partial(pmap,axis_name='x')    
                @jit    
                def advance_one_step(field,dx,dy,dt):
                    field = advance_flux(field,dx,dy,dt)
                    return field
            else:
                @jit    
                def advance_one_step(field,dx,dy,dt):
                    field = advance_flux(field,dx,dy,dt)
                    return field
    
    
    return advance_one_step,rhs
        

    


