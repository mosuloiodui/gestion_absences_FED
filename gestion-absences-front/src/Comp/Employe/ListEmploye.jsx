import React, { useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import {fetchEmployes,deleteEmploye} from '../../Features/Employe/EmployeSlice';
import { setSelectedEmploye } from '../../Features/Employe/EmployeSlice'; 
import {useNavigate} from 'react-router-dom';

export default function ListEmploye() {

const dispatch = useDispatch();
const data = useSelector((state) => state.employes.list);
const status = useSelector((state) => state.employes.status);
const error = useSelector((state) => state.employes.error);
const navigate = useNavigate(); 

	  
useEffect(() => {
	if (status === 'idle') {
		dispatch(fetchEmployes());
		}
}, [dispatch, status]);

const handleEdit = async(employe) => {
	dispatch(setSelectedEmploye(employe));
	navigate('/editEmploye');
};
  

const handleDelete = async(id) => {
	if (window.confirm("Êtes-vous sûr de vouloir supprimer cet employé ?")) {
	await dispatch(deleteEmploye(id)); 
	dispatch(fetchData()); 
	}
  };

  if (status === 'loading') {
	return <div>Loading...</div>;
	}
	  
if (status === 'failed') {
	return <div>Error: {error}</div>;
}

  return (
    
    <div className = "container" >
    <h5 style={{margin: '25px 0px 30px 20px'}}>Liste des Employés</h5>
    	<table className="table table-striped" style={{width:'100%'}}>
    		<thead>
    			<tr>
    				<th>PPR</th>
    				<th>Nom</th>
    				<th>Prenom</th>
    				<th>CNIE</th>
    				<th>Direction</th>
    				<th>Affectation</th>
    				<th>Actions</th>				
    			</tr>		
    		</thead>
    		<tbody>
                {
					Array.isArray(data) && data.length > 0 ?(
                    data.map(employe =>
    			    <tr key ={employe.employeId}>
    				    <td>{employe.ppr}</td>
    				    <td>{employe.nom}</td>
    			        <td>{employe.prenom}</td>
                        <td>{employe.cnie}</td>
                        <td>{employe.direction.directionLibelle}</td>
                        <td>{employe.affectation}</td>
    				    <td style ={{whiteSpace:'nowrap'}}>
    					    <button onClick={() => handleEdit(employe)} className="btn btn-primary btn-sm"> Modifier</button>		 					
    					    <button onClick={() => handleDelete(employe.employeId)} className="btn btn-primary btn-sm" > Supprimer</button> 		
    				    </td>
    			    </tr>)):
					<p></p>
                }
    		</tbody>
    	
    	</table>
    </div>

  )
}
