import React, { useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import { Link , useNavigate} from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import { fetchDirections} from '../../Features/Employe/DirectionSlice';
import {addEmploye} from '../../Features/Employe/EmployeSlice';


const EmployeeForm = () => {
  
    const dispatch = useDispatch();
    const { directions} = useSelector((state) => state.direction);
    const { register, handleSubmit, watch, reset } = useForm();
    const [selectedDirection, setSelectedDirection] = useState(null);
    const navigate = useNavigate(); 


    useEffect(() => {
        dispatch(fetchDirections());
    }, [dispatch]);

    const selectedDirectionId = watch('direction');

    const handleDirectionChange = (e) => {
      const selectedId = parseInt(e.target.value);
      const selectedOption = directions.find(
        (direction) => direction.directionId === selectedId
      );
      setSelectedDirection(selectedOption);
    };

    const onSubmit = async (data) => {
      try {

        const formData = {
          ...data,
          direction: selectedDirection,
        };
        console.log('Form data:', formData);
        await dispatch(addEmploye(formData));
        reset();
        setSelectedDirection(null); 
        if (window.confirm("l'employé a été ajouté avec succès!")) {
          navigate('/');
          }
      } catch (error) {
        console.error('Error saving employee:', error);
      }
    };

    return (
      <div className = "container">
    	<div className = "row">
    	<div className = "col-md-8 mx-auto rounded border p-4 m-4">
    		<h2 class = "text-center mb-5">Ajouter un Employé</h2>
        <form onSubmit={handleSubmit(onSubmit)}>

        <div className = "row mb-3">
            <label className="col-sm-4 col-form-label">PPR:<span className="text-danger">*</span></label>
              <div className="col-sm-8">
                <input className="form-control"{...register('ppr')} required />
              </div>
        </div>

        <div className = "row mb-3">
            <label className="col-sm-4 col-form-label">CNIE:<span className="text-danger">*</span></label>
              <div className="col-sm-8">
                <input className="form-control"{...register('cnie')} required />
              </div>
        </div>

        <div className = "row mb-3">
            <label className="col-sm-4 col-form-label">Nom:<span className="text-danger">*</span></label>
              <div className="col-sm-8">
                <input className="form-control"{...register('nom')} required />
              </div>
        </div>

        <div className = "row mb-3">
            <label className="col-sm-4 col-form-label">Prenom:<span className="text-danger">*</span></label>
              <div className="col-sm-8">
                <input className="form-control"{...register('prenom')} required />
              </div>
        </div>

         <div className = "row mb-3">
    					<label className="col-sm-4 col-form-label">Direction<span className="text-danger">*</span></label>
    					<div className="col-sm-8">
    						<select {...register('direction')} required className="form-control" 
                onChange={handleDirectionChange} >
    							<option value ="">-- Sélectionner une Direction --</option>
    							{directions.map((direction) => (
                    <option key={direction.directionId} value={direction.directionId}>
                        {direction.directionLibelle}
                    </option>
                ))}
    						</select>		
    					</div>
    		</div>
        <div className = "row mb-3">
            <label className="col-sm-4 col-form-label">Affectation:<span className="text-danger">*</span></label>
              <div className="col-sm-8">
                <input className="form-control"{...register('affectation')} required />
              </div>
        </div>
            <div class="row">
    					<div class = "offset-sm-4 col-sm-4 d-grid">
    						<button type = "submit" class = "btn btn-primary"> Enregistrer </button>
    					</div>
    					<div class = "col-sm-4 d-grid">
    						<Link to ='/'class="btn btn-outline-primary" role="button"> Annuler </Link>
    					</div>
    				</div>
        </form>
      </div>
    </div>
    </div>
    );
};

export default EmployeeForm;
